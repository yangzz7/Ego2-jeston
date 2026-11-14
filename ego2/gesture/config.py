"""Gesture settings; reuse calibration and input configuration from ego2."""
from pathlib import Path
import math

from ..common.config import load_profile, resolve_path


from ..common.paths import CONFIGS

DEFAULT_CONFIG = CONFIGS / "gesture.yaml"
DEFAULTS = {
    "backend": "cpu",
    "model": "../models/gesture_recognizer.task",
    "num_hands": 2,
    "detection_confidence": 0.5,
    "presence_confidence": 0.5,
    "tracking_confidence": 0.5,
    "gesture_confidence": 0.6,
    "handedness_confidence": 0.65,
    "handedness_confirm_ms": 250.0,
    "handedness_initial_window_ms": 1000.0,
    "handedness_switch_confidence": 0.85,
    "handedness_switch_ms": 500.0,
    "smoothing_ms": 35.0,
    "confirm_ms": 120.0,
    "track_timeout_ms": 300.0,
    "track_reacquire_ms": 650.0,
    "association_distance": 0.18,
    "pinch_enter": 0.25,
    "pinch_exit": 0.38,
    "released_ms": 250.0,
}
VIEW_DEFAULTS = {"width": 640, "height": 640, "hfov": 105.0, "yaw": 0.0, "pitch": -15.0}


def load_config(path=DEFAULT_CONFIG, *, sn=None):
    profile, cameras = load_profile(path, sn=sn)
    settings = {**DEFAULTS, **profile.get("gesture", {})}
    if settings["backend"] not in ("cpu", "gpu", "auto"):
        raise ValueError("gesture.backend must be cpu, gpu or auto")
    settings["model"] = str(resolve_path(profile["_base"], settings["model"]))
    if settings["num_hands"] not in (1, 2):
        raise ValueError("gesture.num_hands must be 1 or 2")
    for key in ("detection_confidence", "presence_confidence", "tracking_confidence", "gesture_confidence",
                "handedness_confidence", "handedness_switch_confidence"):
        value = float(settings[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"gesture.{key} must be in [0, 1]")
        settings[key] = value
    for key in ("smoothing_ms", "confirm_ms", "track_timeout_ms", "track_reacquire_ms", "released_ms",
                "handedness_confirm_ms", "handedness_initial_window_ms", "handedness_switch_ms"):
        value = float(settings[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"gesture.{key} must be finite and nonnegative")
        settings[key] = value
    if not settings['track_timeout_ms'] <= settings['track_reacquire_ms'] <= 1000:
        raise ValueError('track_reacquire_ms must be >= track_timeout_ms and <= 1000')
    if settings["handedness_initial_window_ms"] < settings["handedness_confirm_ms"]:
        raise ValueError("handedness_initial_window_ms must be >= handedness_confirm_ms")
    if settings["handedness_switch_confidence"] < settings["handedness_confidence"]:
        raise ValueError("handedness_switch_confidence must be >= handedness_confidence")
    if settings["handedness_switch_ms"] < settings["handedness_confirm_ms"]:
        raise ValueError("handedness_switch_ms must be >= handedness_confirm_ms")
    for key in ("association_distance", "pinch_enter", "pinch_exit"):
        settings[key] = float(settings[key])
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"gesture.{key} must be finite and positive")
    if settings["pinch_enter"] >= settings["pinch_exit"]:
        raise ValueError("pinch_enter must be smaller than pinch_exit (hysteresis)")
    profile["gesture"] = settings
    profile["inference_view"] = {**VIEW_DEFAULTS, **profile.get("inference_view", {})}
    profile["viewer"] = {"memory_limit": "1GiB", **profile.get("viewer", {})}
    # Repeated/rewound video timestamps cannot be passed to a temporal model.
    if profile["input"].get("type") == "video" and profile["input"].get("loop", False):
        raise ValueError("Gesture video input requires loop: false; restart to replay")
    profile["input"].setdefault("loop", False)
    return profile, cameras
