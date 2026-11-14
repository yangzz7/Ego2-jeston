"""Recover closed/interrupted gesture recordings into a new dataset directory.

Validated media and raw transport are hardlinked; journals and calibration
snapshots are copied. Missing/corrupt Cyper RGB videos are rebuilt from the
original Bayer bytes with the saved demosaic/white-balance settings. Nothing
is interpolated, and inference gaps remain explicitly missing.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import math
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from ..common.sources import decode_cyper, white_balance
from .dataset import (GestureDataset, VIDEO_KEYS, VIDEO_PATH, _features,
                      _hand_values, _json, _Moments, _Video, _write_json)


def _rows(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as exc:
                raise ValueError(f"Invalid journal {path}, line {number}: {exc}") from exc
            yield row


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Invalid dataset-relative path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes its directory: {relative}")
    return path


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid {name}: expected integer >= {minimum}")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}: expected finite number")
    return value


def _validate_journals(source):
    count, episode, frame, previous_sequence, previous_stamp = 0, 0, 0, -1, -math.inf
    first_stamp = None
    raw_ends, raw_sizes = {}, {}
    for row in _rows(source / "raw/capture.jsonl"):
        try:
            if _integer(row["index"], "index") != count:
                raise ValueError("capture indices are not contiguous")
            current_episode = _integer(row["episode_index"], "episode_index")
            if current_episode == episode + 1 and frame:
                episode, frame = current_episode, 0
            if current_episode != episode or _integer(row["frame_index"], "frame_index") != frame:
                raise ValueError("episode/frame indices are not contiguous")
            sequence = _integer(row["sequence"], "source sequence")
            stamp = _finite(row["source_timestamp"], "source timestamp")
            _finite(row["receive_timestamp"], "receive timestamp")
            capture_time = _finite(row["capture_time"], "capture time")
            if sequence <= previous_sequence or stamp <= previous_stamp:
                raise ValueError("source sequences/timestamps are not strictly increasing")
            if first_stamp is None:
                first_stamp = stamp
            if not math.isclose(capture_time, stamp - first_stamp, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("capture time disagrees with source timestamp")
            if not isinstance(row["metadata"], dict):
                raise ValueError("capture metadata must be an object")
            relative = row["raw_transport_path"]
            offset = _integer(row["raw_transport_offset"], "raw offset")
            length = _integer(row["raw_transport_length"], "raw length")
            if relative:
                path = _inside(source, relative)
                if relative not in raw_sizes:
                    raw_sizes[relative] = path.stat().st_size
                descriptor = row["metadata"]["raw_transport"]
                shape, dtype = descriptor["shape"], np.dtype(descriptor["dtype"])
                if (not isinstance(shape, list) or not shape or dtype.hasobject or
                        any(_integer(n, "raw shape", 1) != n for n in shape)):
                    raise ValueError("invalid raw transport descriptor")
                if descriptor.get("encoding") != "raw" or descriptor.get("order") != "C":
                    raise ValueError("unsupported raw transport encoding/order")
                if length != math.prod(shape) * dtype.itemsize or offset != raw_ends.get(relative, 0):
                    raise ValueError("raw byte ranges disagree with shape or are not contiguous")
                if offset + length > raw_sizes[relative]:
                    raise ValueError(f"truncated raw transport: {relative}")
                raw_ends[relative] = offset + length
            elif offset or length:
                raise ValueError("raw bytes referenced without a raw transport path")
            count, frame = count + 1, frame + 1
            previous_sequence, previous_stamp = sequence, stamp
        except (KeyError, TypeError, OSError, ValueError) as exc:
            raise ValueError(f"Cannot recover capture row {count}: {exc}") from exc
    if not count:
        raise ValueError("Cannot recover a dataset with no captured frames")
    captures = iter(_rows(source / "raw/capture.jsonl"))
    capture = next(captures, None)
    inference_count, previous_sequence = 0, -1
    for result in _rows(source / "raw/inference.jsonl"):
        try:
            sequence = _integer(result["sequence"], "inference sequence")
            if sequence <= previous_sequence:
                raise ValueError("inference sequences are not strictly increasing")
            while capture is not None and capture["sequence"] < sequence:
                capture = next(captures, None)
            if capture is None or capture["sequence"] != sequence:
                raise ValueError("inference references a frame that was not captured")
            _finite(result["stamp"], "inference stamp")
            if _finite(result["processing_ms"], "processing duration") < 0:
                raise ValueError("negative processing duration")
            if not isinstance(result["hands"], list) or len(result["hands"]) != 2:
                raise ValueError("inference must contain two hand lists")
            _hand_values(result["hands"])
            if "detections" not in result:
                raise ValueError("inference detections field missing")
            inference_count += 1
            previous_sequence = sequence
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot recover inference row {inference_count}: {exc}") from exc
    return dict(capture_count=count, inference_count=inference_count, episode_count=episode + 1,
                raw_paths=sorted(raw_ends),
                unjournaled_raw_bytes={p: raw_sizes[p] - end for p, end in raw_ends.items()
                                      if raw_sizes[p] != end})


def _video_stats(path, length, fps, shape=None, expected_hashes=None):
    """Decode every expected frame and check EOF, not just container headers."""
    decoder = cv2.VideoCapture(str(path))
    moments = _Moments()
    try:
        if not decoder.isOpened():
            raise ValueError(f"Cannot decode video: {path}")
        actual_fps = decoder.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(actual_fps) or abs(actual_fps - fps) > max(0.001, fps * 0.001):
            raise ValueError(f"Video fps differs from saved fps: {actual_fps} != {fps}")
        for frame in range(length):
            ok, image = decoder.read()
            if not ok or image is None:
                raise ValueError(f"Video is corrupt/truncated at frame {frame}: {path}")
            if shape is None:
                shape = image.shape
            if tuple(image.shape) != tuple(shape):
                raise ValueError(f"Video image shape differs from saved shape: {path}")
            if expected_hashes is not None and hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest() != expected_hashes[frame]:
                raise ValueError(f"Decoded RGB differs from original capture at frame {frame}: {path}")
            step_y, step_x = max(1, math.ceil(image.shape[0] / 64)), max(1, math.ceil(image.shape[1] / 64))
            moments.add(image[::step_y, ::step_x, ::-1].reshape(-1, 3) / 255.0)
        if decoder.read()[0]:
            raise ValueError(f"Video contains unjournaled extra frames: {path}")
        return moments.stats(image=True, count=length), list(shape)
    finally:
        decoder.release()


def _regenerate_video(source, destination, rows, eye, cfg, fps, ffmpeg, min_free_bytes=0):
    if cfg.get("type") != "cyper":
        raise ValueError("Cannot regenerate corrupt video: saved input is not Cyper Bayer transport")
    width, height = int(cfg.get("width", 1280)), int(cfg.get("height", 1024))
    encoder = None
    try:
        for row in rows:
            if not row["raw_transport_path"]:
                raise ValueError(f"Cannot regenerate corrupt video: raw transport is missing for frame {row['index']} "
                                 "(recording with --no-raw cannot restore corrupt RGB)")
            descriptor = row["metadata"]["raw_transport"]
            if descriptor["shape"] != [height, width, 2] or np.dtype(descriptor["dtype"]) != np.dtype("uint8"):
                raise ValueError(f"Cannot regenerate frame {row['index']}: raw Bayer shape/dtype disagrees with saved profile")
            with _inside(source, row["raw_transport_path"]).open("rb") as handle:
                handle.seek(row["raw_transport_offset"])
                payload = handle.read(row["raw_transport_length"])
            if len(payload) != row["raw_transport_length"]:
                raise ValueError(f"Raw transport changed/truncated during recovery at frame {row['index']}")
            raw = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 2)
            images, _ = decode_cyper(raw, width, height, cfg.get("bayer", "auto"))
            image = white_balance(images[eye]) if cfg.get("white_balance", True) else images[eye]
            hashes = row['metadata'].get('deferred_rgb_sha256')
            if hashes is not None and hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest() != hashes[eye]:
                raise ValueError(f"RAW reconstruction differs from original RGB at frame {row['index']}, eye {eye}")
            if encoder is None:
                encoder = _Video(destination, image.shape, fps, ffmpeg, isolate_signals=True)
            if shutil.disk_usage(destination.parent).free < min_free_bytes + 2 * image.nbytes:
                raise RuntimeError('Low disk space while encoding recovered video; original RAW retained')
            encoder.write(image)
    finally:
        if encoder is not None:
            encoder.close()


def _hardlink(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        raise OSError(f"Cannot hardlink {source} to {destination}; choose a new destination on the same filesystem") from exc


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover_dataset(source, destination):
    """Return a recovery report; destination must be new, source stays untouched.

    Run only after recording has stopped. Media/raw hardlinks share immutable
    file contents with the source; the recovered directory must stay archival.
    Any failure leaves a clearly marked incomplete destination for diagnosis.
    """
    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    if destination == source or destination.is_relative_to(source) or destination.exists():
        raise FileExistsError("Recovery destination must be a new directory outside the source dataset")
    metadata = json.loads((source / "meta/ego2.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or not isinstance(metadata.get("profile"), dict):
        raise ValueError("Unsupported Ego2 metadata or missing saved profile")
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    cfg = metadata["profile"].get("input", {})
    fps = _finite(info.get("fps", metadata.get("fps", cfg.get("fps"))), "saved fps")
    if fps <= 0:
        raise ValueError("Saved fps must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Recovery requires ffmpeg with libx264rgb in PATH")
    counts = _validate_journals(source)
    task = metadata.get("task", "gesture capture")
    tasks_path = source / "meta/tasks.parquet"
    if tasks_path.exists():
        import pandas as pd
        tasks = pd.read_parquet(tasks_path)
        if len(tasks) != 1 or tasks.iloc[0]["task_index"] != 0:
            raise ValueError("Recovery requires one gesture capture task")
        task = tasks.index[0]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Invalid saved capture task")
    features = _features()
    recovery = dict(source=str(source), recovered_at=datetime.now(timezone.utc).isoformat(),
                    source_recording_status=metadata.get("recording_status"),
                    source_error=metadata.get("error"), source_finalized_frames=info.get("total_frames", 0),
                    capture_count=counts["capture_count"], inference_count=counts["inference_count"],
                    missing_inference_count=counts["capture_count"] - counts["inference_count"],
                    unjournaled_raw_bytes=counts["unjournaled_raw_bytes"],
                    media_storage="hardlinks for valid media/raw; new files for regenerated videos",
                    journal_sha256={}, videos=[])
    extra = deepcopy(metadata)
    extra.pop("error", None)
    extra.update(recording_status="recovering", recovery=recovery)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "meta").mkdir()
    (destination / "raw").mkdir()
    _write_json(destination / "meta/ego2.json", extra)
    try:
        for name in ("capture.jsonl", "inference.jsonl"):
            relative = "raw/" + name
            shutil.copyfile(source / relative, destination / relative)
            recovery["journal_sha256"][relative] = _digest(destination / relative)
        # Preserve original segment evidence separately; the active journal is rebuilt.
        if (source / "raw/segments.jsonl").exists():
            shutil.copyfile(source / "raw/segments.jsonl", destination / "raw/source_segments.jsonl")
        _write_json(destination / "meta/source_ego2.json", metadata)
        for relative in metadata.get("calibration_files", {}).values():
            path = _inside(source, relative)
            target = _inside(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for relative in counts["raw_paths"]:
            _hardlink(_inside(source, relative), _inside(destination, relative))
        groups = groupby(_rows(destination / "raw/capture.jsonl"), key=lambda row: row["episode_index"])
        with (destination / "raw/segments.jsonl").open("w", encoding="utf-8") as journal:
            for episode, iterator in groups:
                rows = list(iterator)
                chunk, file = divmod(episode, 1000)
                video_stats = {}
                for eye, key in enumerate(VIDEO_KEYS):
                    relative = VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
                    original, target = _inside(source, relative), _inside(destination, relative)
                    shape = features.get(key, info.get("features", {}).get(key, {})).get("shape")
                    if shape is None and cfg.get("type") == "cyper":
                        shape = [int(cfg.get("height", 1024)), int(cfg.get("width", 1280)), 3]
                    detail = dict(path=relative, frames=len(rows))
                    expected_hashes = None
                    if metadata.get('deferred_video'):
                        expected_hashes = []
                        for row in rows:
                            hashes = row['metadata'].get('deferred_rgb_sha256')
                            if not isinstance(hashes, list) or len(hashes) != 2 or any(
                                    not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
                                raise ValueError('Deferred recording is missing original RGB checksums')
                            expected_hashes.append(hashes[eye])
                    try:
                        if not original.is_file():
                            raise ValueError('Video has not been encoded yet')
                        stats, shape = _video_stats(original, len(rows), fps, shape, expected_hashes)
                    except ValueError as exc:
                        detail.update(action="regenerated_from_raw", original_error=str(exc))
                        _regenerate_video(source, target, rows, eye, cfg, fps, ffmpeg,
                                          int(metadata.get('min_free_bytes', 0)))
                        stats, shape = _video_stats(target, len(rows), fps, shape, expected_hashes)
                    else:
                        _hardlink(original, target)
                        detail["action"] = "validated_and_hardlinked"
                    features[key] = dict(dtype="video", shape=shape, names=["height", "width", "channels"],
                        info={"video.height": shape[0], "video.width": shape[1], "video.codec": "h264",
                              "video.pix_fmt": "gbrp", "video.is_depth_map": False, "video.fps": fps,
                              "video.channels": 3, "has_audio": False})
                    video_stats[key] = stats
                    recovery["videos"].append(detail)
                journal.write(_json(dict(episode_index=episode, length=len(rows),
                    dataset_from_index=rows[0]["index"], dataset_to_index=rows[-1]["index"] + 1,
                    video_stats=video_stats)) + "\n")
        if extra.get('deferred_video'):
            extra['deferred_video'] = dict(extra['deferred_video'], materialized=True)
        GestureDataset.finalize_journals(destination, fps=fps, task=task, features=features, extra=extra,
            capture_count=counts["capture_count"], inference_count=counts["inference_count"],
            episode_count=counts["episode_count"])
    except BaseException as exc:
        extra.update(recording_status="recovery_failed", error=str(exc) or type(exc).__name__)
        _write_json(destination / "meta/ego2.json", extra)
        raise
    return recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(recover_dataset(args.source, args.destination), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
