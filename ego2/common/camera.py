"""Select one camera's identity and calibration together for all entry points."""
from pathlib import Path
import copy
import re
import yaml

from .paths import ROOT

CAMERAS = ROOT / "cameras"


def add_camera_argument(parser):
    parser.add_argument("--sn", metavar="SN", help=(
        "Select cameras/<SN>/camera.yaml and the USB camera with that serial; "
        "default: camera in --config (bundled configs: s200032)"))


def validate_camera_args(parser, args):
    if args.sn is None:
        return
    if getattr(args, "device", None):
        parser.error("--sn cannot be combined with --device; serial selection discovers the device automatically")
    if getattr(args, "replay", None) or getattr(args, "command", None) == "export":
        parser.error("--sn is for new inputs; replay/export uses the recording's preserved calibration")


def print_camera_selection(cfg):
    serial = cfg.get("serial", cfg.get("input", {}).get("serial"))
    calibration = cfg["calibration"]
    if isinstance(calibration, dict):
        calibration = calibration["file"]
    print(f"[camera] SN={serial} | calibration={calibration}", flush=True)


def load_camera(path=None, *, sn=None):
    if sn is not None:
        if not isinstance(sn, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sn):
            raise ValueError("Invalid --sn: use the USB serial number, not a file path")
        path = CAMERAS / sn / "camera.yaml"
        if not path.is_file():
            available = ", ".join(p.parent.name for p in sorted(CAMERAS.glob("*/camera.yaml"))) or "none"
            raise ValueError(f"No camera profile for SN {sn}: {path}; configured SNs: {available}. "
                             "Add this camera's camera.yaml and calibration files first.")
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        bundle = yaml.safe_load(f)
    if not isinstance(bundle,dict) or not bundle.get("serial"):
        raise ValueError(f"Invalid camera bundle: {path}")
    if sn is not None and bundle["serial"] != sn:
        raise ValueError(f"Camera profile serial {bundle['serial']!r} does not match requested SN {sn!r}: {path}")
    bundle = copy.deepcopy(bundle)
    for key in ("calibration","imu_calibration"):
        p = Path(bundle[key]).expanduser()
        bundle[key] = str((p if p.is_absolute() else path.parent/p).resolve())
        if not Path(bundle[key]).is_file():
            raise ValueError(f"Missing camera parameter file: {bundle[key]}")
    bundle["input"]["serial"] = bundle["serial"]
    bundle["path"] = str(path)
    return bundle
