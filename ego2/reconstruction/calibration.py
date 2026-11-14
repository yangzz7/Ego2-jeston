"""One explicit calibration convention for ORB, rectification and export."""
from pathlib import Path
import math
import json

import cv2
import numpy as np

from ..common.config import load_calibration, read_yaml, resolve_path, se3

from ..common.paths import CONFIGS
from ..common.camera import load_camera

DEFAULT_CONFIG = CONFIGS / "reconstruction.yaml"


def load_config(path=DEFAULT_CONFIG, *, sn=None):
    path = Path(path).resolve()
    cfg = read_yaml(path)
    bundle = load_camera(resolve_path(path.parent,cfg["camera"]), sn=sn)
    cfg["camera"] = bundle["path"]
    for key in ("serial","calibration","imu_calibration","reference_camera","acceleration_scale"):
        cfg[key] = bundle[key]
    for key in ("calibration", "imu_calibration", "orb_source", "vocabulary"):
        cfg[key] = str(resolve_path(path.parent, cfg[key]))
        if not Path(cfg[key]).exists():
            raise ValueError(f"Missing {key}: {cfg[key]}")
    if cfg["reference_camera"] != "cam1":
        raise ValueError("This SDK adapter binds cam1=high byte, cam0=low byte; reference must be cam1")
    if not 1 <= float(cfg["fps"]) <= 30:
        raise ValueError("fps must be in [1, 30]")
    for key in ("acceleration_scale", "max_imu_gap"):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    d = cfg["depth"]
    if not 0 < d["min_m"] < d["max_m"] or d["voxel_m"] <= 0:
        raise ValueError("Invalid metric depth/voxel limits")
    if d["num_disparities"] <= 0 or d["num_disparities"] % 16:
        raise ValueError("num_disparities must be a positive multiple of 16")
    if d["block_size"] < 3 or d["block_size"] % 2 == 0:
        raise ValueError("block_size must be odd and >= 3")
    if min(d["sample_stride"], d["max_live_keyframes"], d["height"]) < 1 or d["width"] <= d["num_disparities"]:
        raise ValueError("Invalid depth dimensions or live point cloud limits")
    radius = float(cfg["viewer"].get("point_radius_px",0.65))
    if not math.isfinite(radius) or not 0 < radius <= 5:
        raise ValueError("point_radius_px must be in (0, 5]")
    cfg["config_path"] = str(path)
    return cfg


def calibration(cfg):
    cams = load_calibration(cfg["calibration"])
    left, right = cams["cam1"], cams["cam0"]
    if left.size != right.size or left.size != (1280, 1024):
        raise ValueError("Cyperstereo capture requires 1280 x 1024 calibration for both cameras")
    # T_right_left takes a cam1 point into cam0 coordinates.
    T_right_left = right.T_from_root @ np.linalg.inv(left.T_from_root)
    return left, right, T_right_left


def write_orb_settings(cfg, path):
    left, right, Trl = calibration(cfg)
    raw = read_yaml(cfg["calibration"])[left.name]
    imu = read_yaml(cfg["imu_calibration"])["imu0"]
    # OpenCV 4.13 no longer writes dotted YAML keys, but ORB's legacy
    # reader still requires them. Emit this small fixed schema explicitly.
    lines = ["%YAML:1.0", "---"]
    def write(key, value):
        if isinstance(value, np.ndarray):
            lines.extend([f"{key}: !!opencv-matrix", f"  rows: {value.shape[0]}",
                          f"  cols: {value.shape[1]}", "  dt: f",
                          "  data: " + json.dumps(value.ravel().tolist())])
        else:
            lines.append(f"{key}: {json.dumps(value)}")
    write("Camera.type", "KannalaBrandt8")
    for prefix, cam in (("Camera", left), ("Camera2", right)):
        values = dict(zip(("fx", "fy", "cx", "cy"),
                          (cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2])))
        values.update(dict(zip(("k1", "k2", "k3", "k4"), cam.D)))
        for key, value in values.items():
            write(prefix + "." + key, float(value))
        write(prefix + ".lappingBegin", 0)
        write(prefix + ".lappingEnd", left.size[0] - 1)
    write("Tlr", np.linalg.inv(Trl)[:3].astype(np.float32))
    # ORB Tbc maps camera -> body. Kalibr T_cam_imu maps body -> camera.
    write("Tbc", np.linalg.inv(se3(raw["T_cam_imu"])).astype(np.float32))
    values = {
        "Camera.width": left.size[0], "Camera.height": left.size[1],
        "Camera.fps": float(cfg["fps"]), "Camera.RGB": 0,
        "Camera.bf": float(left.K[0, 0] * np.linalg.norm(Trl[:3, 3])), "ThDepth": 60.0,
        "IMU.NoiseGyro": float(imu["gyroscope_noise_density"]),
        "IMU.NoiseAcc": float(imu["accelerometer_noise_density"]),
        "IMU.GyroWalk": float(imu["gyroscope_random_walk"]),
        "IMU.AccWalk": float(imu["accelerometer_random_walk"]),
        "IMU.Frequency": int(imu["update_rate"]),
        "ORBextractor.nFeatures": int(cfg["features"]), "ORBextractor.scaleFactor": 1.2,
        "ORBextractor.nLevels": 8, "ORBextractor.iniThFAST": 15, "ORBextractor.minThFAST": 7,
        "Viewer.KeyFrameSize": 0.05, "Viewer.KeyFrameLineWidth": 1.0,
        "Viewer.GraphLineWidth": 0.9, "Viewer.PointSize": 2.0, "Viewer.CameraSize": 0.08,
        "Viewer.CameraLineWidth": 3.0, "Viewer.ViewpointX": 0.0, "Viewer.ViewpointY": -0.7,
        "Viewer.ViewpointZ": -3.5, "Viewer.ViewpointF": 500.0,
    }
    for key, value in values.items():
        write(key, value)
    Path(path).write_text("\n".join(lines)+"\n")
    # Kalibr convention: t_imu = t_camera + timeshift_cam_imu.
    return float(raw["timeshift_cam_imu"])
