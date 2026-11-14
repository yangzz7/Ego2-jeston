"""Marker identity, metric size, camera bundle and explicit coordinate frames."""
import re
from pathlib import Path
import cv2
import numpy as np
from ..common.config import load_profile, read_yaml, se3, resolve_path
from .bundle import marker_ids
from ..common.geometry import View
from ..common.paths import CONFIGS

DEFAULT_CONFIG = CONFIGS / "hand_pose.yaml"


def dictionary(name):
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError("OpenCV >= 4.8 with cv2.aruco is required; install requirements.txt")
    if not isinstance(name, str) or not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown OpenCV marker dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def load_config(path=DEFAULT_CONFIG, *, sn=None):
    cfg, cameras = load_profile(path, sn=sn)
    if "marker_config" in cfg:
        marker_path = resolve_path(Path(path).resolve().parent, cfg["marker_config"])
        markers = read_yaml(marker_path)
        keys = ("markers_configured", "dictionary", "border_bits", "targets")
        missing = [key for key in keys if key not in markers]
        if missing:
            raise ValueError(f"Wrist marker configuration {marker_path} is missing: {', '.join(missing)}")
        # Inline fields remain supported and override the shared marker defaults.
        for key in keys:
            cfg.setdefault(key, markers[key])
        cfg["marker_config"] = str(marker_path)
    if cfg.get("bundle_calibration"):
        cfg["bundle_calibration"] = str(resolve_path(Path(path).resolve().parent, cfg["bundle_calibration"]))
    elif "camera" in cfg:
        cfg["bundle_calibration"] = str(Path(cfg["camera"]).parent / "wrist_bundles.json")
    validate(cfg)
    # The current acquisition adapter uses the native calibrated pixel grid.
    for channel in cfg["channels"]:
        if any(channel.get(key) for key in ("rotate_cw", "flip_horizontal", "flip_vertical", "calibration_crop")):
            raise ValueError("Hand pose currently requires native, unrotated, uncropped calibrated images")
    return cfg, cameras


def validate(cfg):
    d = dictionary(cfg["dictionary"])
    if type(cfg["border_bits"]) is not int or not 1 <= cfg["border_bits"] <= 3:
        raise ValueError("border_bits must be 1, 2 or 3")
    targets = cfg["targets"]
    if not 1 <= len(targets) <= 2:
        raise ValueError("Configure one or two wrist targets")
    ids = [i for target in targets for i in marker_ids(target)]
    if len(set(ids)) != len(ids) or len({t["name"] for t in targets}) != len(targets):
        raise ValueError("Marker IDs must be unique within and across both wrists")
    for target in targets:
        if "id" in target and "ids" in target:
            raise ValueError("Use id for a single tag OR ids for a wrist bundle")
        if "ids" in target and not 5 <= len(target["ids"]) <= 20:
            raise ValueError("A wrist loop requires 5..20 physically ordered tags")
        if any(type(i) is not int or not 0 <= i < len(d.bytesList) for i in marker_ids(target)):
            raise ValueError("Marker ID outside the configured dictionary")
        if not np.isfinite(target.get("palm_offset_deg", 0)):
            raise ValueError("palm_offset_deg must be finite")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", target["name"]):
            raise ValueError("Target names must be simple ASCII names such as left/right")
        if not np.isfinite(target["size_m"]) or not 0.005 <= target["size_m"] <= 0.5:
            raise ValueError("Marker detection-corner edge length must be 0.005..0.5 metres")
        offset = "T_wrist_endpoint" if "ids" in target else "T_marker_endpoint"
        target[offset] = se3(target.get(offset, np.eye(4))).tolist()
        color = target["color"]
        if len(color) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in color):
            raise ValueError("Target color must be three RGB bytes")
    View(**cfg["detection_view"])
    for group in ("quality", "tracking"):
        for key, value in cfg[group].items():
            if isinstance(value, bool):
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{group}.{key} must be finite and nonnegative")
    q, t = cfg["quality"], cfg["tracking"]
    if not isinstance(q["require_stereo"], bool):
        raise ValueError("require_stereo must be a boolean")
    if not 0 < q["min_distance_m"] < q["max_distance_m"] or min(q["min_side_px"], q["max_reprojection_px"]) <= 0:
        raise ValueError("Invalid depth or corner quality limits")
    if min(t["max_gap_s"], t["max_speed_m_s"], t["max_angular_speed_deg_s"], t["pose_history_interval_s"]) <= 0:
        raise ValueError("Time and velocity limits must be positive")
    for key in ("trajectory_points", "pose_history_count"):
        if not isinstance(t[key], int) or not 2 <= t[key] <= 20000:
            raise ValueError(f"{key} must be an integer in 2..20000")
    axis = cfg["viewer"]["axis_length_m"]
    if not np.isfinite(axis) or not 0 < axis <= 0.5:
        raise ValueError("axis_length_m must be in (0, 0.5]")

    for group in ("bundle_fusion", "bundle_calibration_quality"):
        for key, value in cfg.get(group, {}).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{group}.{key} must be positive and finite")

    if any("ids" in target for target in targets):
        if not cfg.get("bundle_calibration"):
            raise ValueError("Multi-tag wrists require a bundle_calibration path")
        calibration = cfg["bundle_calibration_quality"]
        if type(calibration["min_samples"]) is not int or not 3 <= calibration["min_samples"] <= 160:
            raise ValueError("Calibration min_samples must be an integer in 3..160")
