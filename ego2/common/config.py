"""Calibration and profile loading. All paths are relative to their YAML file."""
from dataclasses import dataclass
from pathlib import Path
import copy

import cv2
import numpy as np
import yaml


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def resolve_path(base, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base) / p).resolve()


def se3(value):
    t = np.asarray(value, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Extrinsics must be a finite 4x4 matrix")
    r = t[:3, :3]
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous transform bottom row")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or abs(np.linalg.det(r)-1) > 1e-4:
        raise ValueError("Extrinsic rotation must be orthonormal with determinant +1")
    return t


@dataclass
class Camera:
    name: str
    K: np.ndarray
    D: np.ndarray
    size: tuple
    T_from_root: np.ndarray
    theta_limit: float


def camera(name, block, transform, fmt):
    if fmt == "kalibr":
        if (block.get("camera_model"), block.get("distortion_model")) != ("pinhole", "equidistant"):
            raise ValueError(f"{name}: only pinhole + equidistant is supported; other models need an adapter")
        fx, fy, cx, cy = block["intrinsics"]
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    else:
        if block.get("model") != "opencv_fisheye":
            raise ValueError(f"{name}: model must be opencv_fisheye")
        k = np.asarray(block["K"], np.float64)
    d = np.asarray(block.get("distortion_coeffs", block.get("D")), np.float64).reshape(-1)
    size = tuple(int(x) for x in block["resolution"])
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError(f"{name}: invalid intrinsic matrix")
    if not np.allclose(k[2], [0, 0, 1]) or abs(k[1, 0]) > 1e-9:
        raise ValueError(f"{name}: unsupported intrinsic matrix structure")
    if len(d) != 4 or not np.isfinite(d).all() or len(size) != 2 or min(size) < 2:
        raise ValueError(f"{name}: expected four fisheye coefficients and [width, height]")
    # Stop at the first non-monotonic angle: never fold an extrapolated lens
    # polynomial back into the image. atan2 projection also supports >90 deg rays.
    roots = np.roots([9*d[3], 0, 7*d[2], 0, 5*d[1], 0, 3*d[0], 0, 1])
    limits = [z.real for z in roots if abs(z.imag) < 1e-7 and z.real > 0]
    limit = min([np.pi - 1e-5] + limits) - 1e-6
    if "max_theta_deg" in block:
        limit = min(limit, np.deg2rad(float(block["max_theta_deg"])))
    if limit <= 0:
        raise ValueError(f"{name}: no valid angular domain")
    return Camera(name, k, d, size, se3(transform), limit)


def load_calibration(path, fmt="kalibr"):
    data = read_yaml(path)
    if fmt == "kalibr":
        names = sorted((n for n in data if n.startswith("cam") and n[3:].isdigit()), key=lambda n: int(n[3:]))
        if names != [f"cam{i}" for i in range(len(names))] or len(names) < 2:
            raise ValueError("Kalibr chain must contain consecutive cam0, cam1, ...")
        out, transform = {}, np.eye(4)
        for i, name in enumerate(names):
            if i:
                transform = se3(data[name]["T_cn_cnm1"]) @ transform
            out[name] = camera(name, data[name], transform, fmt)
        return out
    if fmt == "opencv":
        blocks = data["cameras"]
        if len(blocks) != 2:
            raise ValueError("OpenCV YAML format expects exactly two cameras")
        names = list(blocks)
        # Explicit name binding avoids relying on YAML insertion order.
        ext = data["extrinsics"]
        a, b = ext["from"], ext["to"]
        if a == b or set([a, b]) != set(names):
            raise ValueError("extrinsics.from/to must identify the two cameras")
        return {a: camera(a, blocks[a], np.eye(4), fmt),
                b: camera(b, blocks[b], se3(ext["matrix"]), fmt)}
    raise ValueError(f"Unknown calibration format: {fmt}")


DEFAULT_DISPLAY = dict(width=1280, height=720, hfov=110.0, yaw=0.0,
                       pitch=0.0, distance=2.0, blend_width=16,
                       coverage_mode="free", min_coverage=0.999)
DEFAULT_FUSION = dict(seam_scale=0.25, seam_interval=0.2,
                      seam_band=[0.62, 0.85], seam_preference=0.74,
                      temporal_weight=0.12, gain_tau=1.0,
                      gain_limits=[0.8, 1.25], mismatch_threshold=35.0)


def load_profile(path, calibration_override=None, *, sn=None):
    path = Path(path).resolve()
    p = copy.deepcopy(read_yaml(path))
    if "camera" in p or sn is not None:
        from .camera import load_camera
        bundle = load_camera(resolve_path(path.parent, p["camera"]) if "camera" in p else None, sn=sn)
        p["camera"] = bundle["path"]
        p["calibration"] = {"file":bundle["calibration"],"format":"kalibr"}
        p["input"] = {**bundle["input"],**p.get("input",{})}
        p["input"].setdefault("acceleration_scale", bundle.get("acceleration_scale", 9.80665))
        if sn is not None:
            # A profile's stale device/SN or sensor geometry must never select
            # another unit while using the requested camera's calibration.
            for key in ("type", "serial", "width", "height", "bayer"):
                if key in bundle["input"]:
                    p["input"][key] = bundle["input"][key]
            p["input"]["device"] = "auto"
            p["input"]["acceleration_scale"] = bundle.get("acceleration_scale", 9.80665)
        p["channels"] = copy.deepcopy(bundle["channels"])
    p["_path"] = str(path)
    p["_base"] = str(path.parent)
    p["calibration"]["file"] = str(Path(calibration_override).expanduser().resolve() if calibration_override else resolve_path(path.parent, p["calibration"]["file"]))
    cams = load_calibration(p["calibration"]["file"], p["calibration"].get("format", "kalibr"))
    channels = p.get("channels", [])
    if len(channels) != 2 or len({c["camera"] for c in channels}) != 2:
        raise ValueError("Provide two channels with distinct calibration camera names")
    for c in channels:
        if c["camera"] not in cams:
            raise ValueError(f"Unknown calibration camera: {c['camera']}")
        if c.get("rotate_cw", 0) not in (0, 90, 180, 270):
            raise ValueError("rotate_cw must be 0, 90, 180 or 270")
        if c.get("valid_mask"):
            c["valid_mask"] = str(resolve_path(path.parent, c["valid_mask"]))
    p["reference"] = p.get("reference", channels[0]["camera"])
    if p["reference"] not in [c["camera"] for c in channels]:
        raise ValueError("reference must identify one of the two bound cameras")
    p["display"] = {**DEFAULT_DISPLAY, **p.get("display", {})}
    p["fusion"] = {**DEFAULT_FUSION, **p.get("fusion", {})}
    f = p["fusion"]
    lo, hi = f["seam_band"]
    if not 0 <= lo < hi <= 1 or not 0 < float(f["seam_scale"]) <= 1:
        raise ValueError("Invalid seam_band or seam_scale")
    if float(f["seam_interval"]) < 0 or float(f["gain_tau"]) <= 0:
        raise ValueError("Invalid fusion time constants")
    if not 0 < f["gain_limits"][0] <= f["gain_limits"][1]:
        raise ValueError("Invalid gain_limits")
    return p, cams


def orient(image, channel):
    """Undo transport orientation BEFORE mapping into the calibration frame."""
    rotation = channel.get("rotate_cw", 0)
    if rotation:
        image = np.rot90(image, -(rotation // 90))
    if channel.get("flip_horizontal", False):
        image = image[:, ::-1]
    if channel.get("flip_vertical", False):
        image = image[::-1]
    return np.ascontiguousarray(image)
