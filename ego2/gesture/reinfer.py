"""Recompute every saved frame, then atomically replace derived annotations.

This explicitly modifies the supplied dataset. Camera acquisition and Rerun are
never started; the original live annotations and metadata remain as backups.
"""
import argparse
from contextlib import closing
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

from ..common.config import Camera
from .app import Pipeline
from .dataset import GestureDataset, _json
from .replay import DatasetReplay


def _asset(root, value, profile, snapshots):
    """Prefer the recording's snapshot to any subsequently edited live asset."""
    path = Path(value).expanduser()
    original = path if path.is_absolute() else Path(profile.get("_base", ".")) / path
    saved = snapshots.get(str(path)) or snapshots.get(str(original.resolve()))
    if saved:
        candidate = (root / saved).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Saved calibration asset must remain inside the dataset")
    else:
        candidate = original.resolve()
        if not candidate.is_file() and not path.is_absolute():
            candidate = (root / path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing saved inference asset: {candidate}")
    return candidate


def _settings(root, metadata):
    profile = copy.deepcopy(metadata["profile"])
    model = metadata.get("model", {})
    expected = model.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Reinference requires the original recorded model SHA256")
    value = model.get("path") or profile.get("gesture", {}).get("model")
    if not value:
        raise ValueError("Missing recorded inference model path")
    snapshots = metadata.get("calibration_files", {})
    model_path = _asset(root, value, profile, snapshots)
    with model_path.open("rb") as handle:
        # Jetson's Ubuntu 22.04 uses Python 3.10 (file_digest starts at 3.11).
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"Recorded model SHA256 mismatch: {model_path}")
    profile["gesture"]["model"] = str(model_path)
    for channel in profile["channels"]:
        if channel.get("valid_mask"):
            channel["valid_mask"] = str(_asset(root, channel["valid_mask"], profile, snapshots))
    cameras = {}
    for name, values in metadata.get("cameras", {}).items():
        values = dict(values)
        for key in ("K", "D", "T_from_root"):
            values[key] = np.asarray(values[key], dtype=np.float64)
        values["size"] = tuple(values["size"])
        cameras[name] = Camera(**values)
    for channel in profile["channels"]:
        if channel["camera"] not in cameras:
            raise ValueError(f"Missing recorded camera: {channel['camera']}")
    return profile, cameras, actual


def _exchange_directories(stage, root):
    """Linux atomic exchange: readers see either complete generation of tables."""
    library = ctypes.CDLL(None, use_errno=True)
    exchange = getattr(library, "renameat2", None)
    if exchange is None:
        raise RuntimeError("Atomic dataset reinference requires Linux renameat2")
    exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                         ctypes.c_uint]
    exchange.restype = ctypes.c_int
    if exchange(-100, os.fsencode(stage), -100, os.fsencode(root), 2) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(root))


def _link_unchanged(root, stage):
    """Keep the same media bytes/inodes while constructing a complete generation."""
    for source in root.rglob("*"):
        relative = source.relative_to(root)
        # These directories were rebuilt; obsolete derived shards must not leak
        # into the replacement generation.
        if relative.parts[0] == "data" or relative.parts[:2] == ("meta", "episodes"):
            continue
        target = stage / relative
        if target.exists() or target.is_symlink():
            continue
        if source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
        elif source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)


def reinfer_dataset(root):
    """Reinfer all frames in-place, preserving live results and atomic rollback.

    An explicit dataset directory is required. Repeating this operation is
    rejected so that the original live journal can never be overwritten.
    """
    dataset = DatasetReplay(root)
    root = dataset.root
    if dataset.metadata.get("recording_status") != "complete":
        raise ValueError("Recover/finalize the dataset before running reinference")
    if dataset.metadata.get("capture_count") != dataset.total_frames:
        raise ValueError("Reinference requires every captured frame to be finalized")
    for relative in ("raw/inference_live.jsonl", "meta/ego2_live.json", "meta/live"):
        if (root / relative).exists():
            raise FileExistsError(f"Live-result backup already exists: {root / relative}")
    for relative in ("raw/capture.jsonl", "raw/segments.jsonl", "raw/inference.jsonl"):
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Missing reinference journal: {root / relative}")
    profile, cameras, model_hash = _settings(root, dataset.metadata)
    import pandas as pd
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if len(tasks) != 1:
        raise ValueError("Reinference currently requires a single recorded task")
    task = str(tasks.index[0])
    count, previous_inferred = 0, 0
    extra = copy.deepcopy(dataset.metadata)
    started = datetime.now(timezone.utc).isoformat()
    # Same-filesystem staging is required for both hardlinks and atomic rename.
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.reinfer-", dir=root.parent) as temporary:
        stage = Path(temporary)
        (stage / "raw").mkdir()
        (stage / "meta").mkdir()
        for name in ("capture", "segments"):
            os.link(root / f"raw/{name}.jsonl", stage / f"raw/{name}.jsonl")
        os.link(root / "raw/inference.jsonl", stage / "raw/inference_live.jsonl")
        os.link(root / "meta/ego2.json", stage / "meta/ego2_live.json")
        shutil.copytree(root / "meta", stage / "meta/live", copy_function=os.link, symlinks=True)
        with closing(Pipeline(profile, cameras)) as pipeline, \
                closing(dataset.frames()) as frames, \
                (stage / "raw/inference.jsonl").open("w", encoding="utf-8") as journal:
            for pair, _, _, row in frames:
                before = time.monotonic()
                stamp, hands = pipeline.process(pair)
                journal.write(_json(dict(sequence=pair.sequence, stamp=stamp,
                    processing_ms=(time.monotonic() - before) * 1000,
                    hands=hands, detections=pipeline.observations)) + "\n")
                count += 1
                previous_inferred += bool(row["observation.inference_valid"])
                if count % 100 == 0:
                    print(f"[reinfer] processed={count}/{dataset.total_frames}", flush=True)
            journal.flush()
            os.fsync(journal.fileno())
        if count != dataset.total_frames:
            raise ValueError(f"Reinference processed {count} frames; expected {dataset.total_frames}")
        extra["reinference"] = dict(method="offline_sequential_all_frames",
            source_journal="raw/inference_live.jsonl", source_metadata="meta/ego2_live.json",
            source_metadata_directory="meta/live",
            previous_inference_count=previous_inferred, inference_count=count,
            filled_missing_frames=count - previous_inferred, model_sha256=model_hash,
            started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
            note="All frames were recomputed in order to preserve temporal tracking; live results are backed up.")
        GestureDataset.finalize_journals(stage, fps=dataset.fps, task=task,
            features=copy.deepcopy(dataset.info["features"]), extra=extra,
            capture_count=count, inference_count=count, episode_count=dataset.total_episodes)
        _link_unchanged(root, stage)
        # No writes to original indexed files occur before this single commit.
        # After exchange, TemporaryDirectory removes the obsolete hardlinks;
        # the newly published dataset keeps every original media inode alive.
        _exchange_directories(stage, root)
    print(f"[reinfer] inferred={count} filled_missing={count - previous_inferred} dataset={root}", flush=True)
    return extra["reinference"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="dataset to update in-place; use a recovered copy to retain the original")
    args = parser.parse_args(argv)
    try:
        reinfer_dataset(args.dataset)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[reinfer] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
