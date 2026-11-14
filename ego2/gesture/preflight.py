"""Offline recording readiness checks; never open a camera or inference graph.

Only a small, automatically removed FFmpeg probe is written under the system
temporary directory. The proposed recording directory is never created.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import zipfile


DEFAULT_MIN_FREE_BYTES = 2 * 1024 ** 3


def _dependencies(require_viewer):
    # Import OpenCV first: Arrow may otherwise load an older libstdc++ into
    # environments mixing system OpenCV and Conda packages.
    names = ["cv2", "numpy", "yaml", "mediapipe", "zmq", "pyarrow", "pandas"]
    if require_viewer:
        names.append("rerun")
    versions, errors = {}, []
    for name in names:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "importable")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("Missing/unusable recording dependencies: " + "; ".join(errors))
    return dict(versions=versions, inference_graph_initialized=False)


def _destination(dataset):
    requested = Path(dataset).expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"Dataset destination already exists: {requested}")
    target = requested.resolve()
    parent = target.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise NotADirectoryError(f"Dataset parent is not a directory: {parent}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise PermissionError(f"Dataset parent is not writable/searchable: {parent}")
    if hasattr(os, "statvfs") and os.statvfs(parent).f_flag & getattr(os, "ST_RDONLY", 1):
        raise PermissionError(f"Dataset parent filesystem is read-only: {parent}")
    return dict(path=str(target), existing_parent=str(parent),
                parent_writable=True, created=False)


def _configuration(profile):
    from ..common.camera import load_camera
    from ..common.config import load_calibration, read_yaml, resolve_path
    from .rectification import Rectifier

    paths = []
    if profile.get("_path"):
        read_yaml(profile["_path"])
        paths.append(str(Path(profile["_path"]).resolve()))
    calibration = profile["calibration"]
    cameras = load_calibration(calibration["file"], calibration.get("format", "kalibr"))
    paths.append(str(Path(calibration["file"]).resolve()))
    if profile.get("camera"):
        camera_path = resolve_path(profile.get("_base", "."), profile["camera"])
        bundle = load_camera(camera_path)
        read_yaml(bundle["imu_calibration"])
        paths.extend([str(camera_path), bundle["imu_calibration"]])
    channels = profile.get("channels", [])
    if len(channels) != 2 or len({channel["camera"] for channel in channels}) != 2:
        raise ValueError("Configuration must bind two distinct calibrated cameras")
    for channel in channels:
        if channel["camera"] not in cameras:
            raise ValueError(f"Unknown calibration camera: {channel['camera']}")
        Rectifier(cameras[channel["camera"]], channel, profile["inference_view"])
        if channel.get("valid_mask"):
            paths.append(str(Path(channel["valid_mask"]).resolve()))
    cfg = profile["input"]
    fps = float(cfg.get("fps", 30))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Input fps must be finite and positive")
    for key in ("width", "height"):
        value = cfg.get(key, 1280 if key == "width" else 1024)
        if isinstance(value, bool) or int(value) != value or value < 3:
            raise ValueError(f"Input {key} must be an integer >= 3")
    return dict(files=list(dict.fromkeys(paths)), cameras=list(cameras), source_type=cfg.get("type"))


def _model(profile):
    from .model import MODEL_SHA256

    model = Path(profile["gesture"]["model"]).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"Gesture model is missing: {model}")
    digest, size = hashlib.sha256(), 0
    with model.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    if not size:
        raise ValueError(f"Gesture model is empty: {model}")
    # Read the task bundle and verify its ZIP CRCs without initializing
    # MediaPipe (which may otherwise probe graphics contexts).
    try:
        with zipfile.ZipFile(model) as archive:
            expected = {"hand_landmarker.task", "hand_gesture_recognizer.task"}
            if not expected.issubset(archive.namelist()):
                raise ValueError("missing hand landmark/gesture task entries")
            broken = archive.testzip()
            if broken:
                raise ValueError(f"bad ZIP checksum in {broken}")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Unreadable gesture model task bundle {model}: {exc}") from exc
    return dict(path=str(model), size_bytes=size, sha256=digest.hexdigest(),
                official_v1_sha256_matches=digest.hexdigest() == MODEL_SHA256,
                inference_graph_initialized=False)


def _storage(parent, profile, minimum, seconds):
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise ValueError("min_free_bytes must be a nonnegative integer")
    if not math.isfinite(float(seconds)) or seconds < 0:
        raise ValueError("seconds must be finite and nonnegative")
    cfg = profile["input"]
    width, height, fps = int(cfg.get("width", 1280)), int(cfg.get("height", 1024)), float(cfg.get("fps", 30))
    rgb_rate = width * height * 6 * fps
    raw_rate = width * height * 2 * fps if cfg.get("type") == "cyper" and cfg.get("preserve_raw", True) else 0
    estimate = math.ceil((rgb_rate + raw_rate) * seconds * 1.1)
    required = minimum + estimate
    usage = shutil.disk_usage(parent)
    result = dict(free_bytes=usage.free, total_bytes=usage.total, minimum_free_bytes=minimum,
                  requested_seconds=seconds, estimated_recording_bytes=estimate, required_bytes=required,
                  raw_bytes_per_second=raw_rate, uncompressed_rgb_bytes_per_second=rgb_rate,
                  estimate_note="Uncompressed stereo RGB plus enabled raw transport, with 10% allowance; actual compressed video/RRD sizes vary.")
    if usage.free < required:
        raise ValueError(f"Insufficient disk space: {usage.free / 1024**3:.2f} GiB free, "
                         f"{required / 1024**3:.2f} GiB required "
                         f"(reserve {minimum / 1024**3:.2f} GiB + recording estimate)")
    return result


def _ffmpeg():
    executable = shutil.which("ffmpeg")
    if not executable:
        raise FileNotFoundError("ffmpeg is missing from PATH; libx264rgb is required for lossless RGB")
    payload = bytes((index * 31 + index // 11) % 256 for index in range(2 * 16 * 16 * 3))
    with tempfile.TemporaryDirectory(prefix="ego2-preflight-") as directory:
        path = Path(directory) / "probe.mp4"
        command = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "16x16",
                   "-framerate", "30", "-i", "pipe:0", "-an", "-c:v", "libx264rgb",
                   "-crf", "0", "-preset", "ultrafast", "-threads", "1", "-g", "30", "-bf", "0", str(path)]
        encoded = subprocess.run(command, input=payload, capture_output=True, timeout=20)
        if encoded.returncode:
            raise RuntimeError("FFmpeg libx264rgb encode failed: " + encoded.stderr.decode(errors="replace").strip())
        decoded = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-nostdin",
                                  "-threads", "1", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24",
                                  "-threads", "1", "pipe:1"], capture_output=True, timeout=20)
        if decoded.returncode or decoded.stdout != payload:
            raise RuntimeError("FFmpeg libx264rgb roundtrip did not preserve both RGB probe frames exactly: " +
                               decoded.stderr.decode(errors="replace").strip())
        return dict(path=executable, encoder="libx264rgb", probe_frames=2,
                    lossless_roundtrip=True, temporary_bytes=path.stat().st_size)


def _camera_node(profile):
    cfg = profile["input"]
    kind = cfg.get("type")
    if kind not in ("cyper", "uvc"):
        return dict(source_type=kind, applicable=False, opened=False)
    device = cfg.get("device", "auto" if kind == "cyper" else 0)
    if device == "auto":
        from ..common.sources import discover_cyper
        try:
            device = discover_cyper(cfg.get("serial"))
        except (OSError, RuntimeError) as exc:
            raise RuntimeError("Camera node is not uniquely visible in this environment; "
                               "check connection/device access before recording: " + str(exc)) from exc
    elif isinstance(device, int) or str(device).isdigit():
        device = "/dev/video" + str(device)
    node = Path(device)
    if not node.exists():
        raise FileNotFoundError(f"Camera node is not visible in this environment: {node}; "
                                "check connection/device access before recording")
    if not stat.S_ISCHR(node.stat().st_mode):
        raise ValueError(f"Configured camera device is not a character device: {node}")
    if not os.access(node, os.R_OK | os.W_OK):
        raise PermissionError(f"Camera node lacks read/write permission: {node}")
    return dict(device=str(node), present=True, permissions=True, opened=False,
                note="Node discovery only; streaming and USB bandwidth have not been tested")


def preflight(profile, dataset, *, min_free_bytes=DEFAULT_MIN_FREE_BYTES,
              seconds=0.0, require_viewer=True, check_camera=True, require_camera=False):
    """Return JSON-safe readiness checks without modifying the target dataset.

    ``profile`` can be a loaded/overridden gesture profile or its YAML path.
    Failed required checks set ``ok=False``; camera absence is an offline-check
    warning unless ``require_camera=True`` (use immediately before recording).
    CPU model graphs, cameras, viewers and network clients stay closed.
    """
    report = dict(ok=True, dataset=str(Path(dataset).expanduser().absolute()) if dataset is not None else None,
                  checks=[], errors=[], warnings=[])

    def check(name, function, optional=False):
        try:
            details = function()
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            report["checks"].append(dict(name=name, status="warning" if optional else "error", message=message))
            report["warnings" if optional else "errors"].append(f"{name}: {message}")
            if not optional:
                report["ok"] = False
            return None
        report["checks"].append(dict(name=name, status="ok", details=details))
        return details

    check("dependencies", lambda: _dependencies(require_viewer))
    destination = check("destination", lambda: _destination(dataset))
    if not isinstance(profile, dict):
        def load():
            from .config import load_config
            loaded, _ = load_config(profile)
            return loaded
        loaded = check("load_config", load)
        profile = loaded
    if isinstance(profile, dict):
        check("configuration", lambda: _configuration(profile))
        check("model", lambda: _model(profile))
        if destination is not None:
            check("storage", lambda: _storage(destination["existing_parent"], profile, min_free_bytes, seconds))
        if check_camera or require_camera:
            check("camera_node", lambda: _camera_node(profile), optional=not require_camera)
    check("ffmpeg", _ffmpeg)
    return report


def format_report(report):
    """Human-readable result for the recording CLI and standalone command."""
    lines = [f"[preflight] {'PASS' if report['ok'] else 'FAIL'}: {report['dataset']}"]
    for check in report["checks"]:
        detail = check.get("message", "")
        if check["name"] == "storage" and check["status"] == "ok":
            values = check["details"]
            detail = f"{values['free_bytes'] / 1024**3:.2f} GiB free; {values['required_bytes'] / 1024**3:.2f} GiB required"
        elif check["name"] == "ffmpeg" and check["status"] == "ok":
            detail = "libx264rgb encoded/decoded 2 frames losslessly"
        elif check["name"] == "model" and check["status"] == "ok":
            detail = "readable task bundle; SHA256=" + check["details"]["sha256"]
        elif check["name"] == "camera_node" and check["status"] == "ok":
            detail = check["details"].get("device", "not applicable to this source")
        lines.append(f"[preflight] {check['status'].upper()} {check['name']}" + (": " + detail if detail else ""))
    lines.append("[preflight] Camera and model inference were not started; dataset directory was not created.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/gesture.yaml")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        parser.error("--min-free-gb must be finite and nonnegative")
    report = preflight(args.config, args.dataset, min_free_bytes=int(args.min_free_gb * 1024**3),
                       seconds=args.seconds, require_viewer=not args.no_viz)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
