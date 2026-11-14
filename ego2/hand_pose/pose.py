"""Metric marker pose from calibrated virtual views; all transforms are SE(3)."""
from dataclasses import dataclass
import cv2
import numpy as np


def corners(size):
    # OpenCV IPPE_SQUARE: canonical decoded TL, TR, BR, BL.
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def transform(rotation_vector, translation):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rotation_vector, np.float64))[0]
    result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def rotation_distance(a, b):
    return float(np.linalg.norm(cv2.Rodrigues(a.T @ b)[0]))


def quaternion_xyzw(rotation):
    vector = cv2.Rodrigues(rotation)[0].ravel()
    angle = np.linalg.norm(vector)
    xyz = vector * (np.sin(angle/2)/angle if angle > 1e-10 else 0.5)
    return np.r_[xyz, np.cos(angle/2)]


@dataclass
class Observation:
    camera: str
    pixels: np.ndarray
    K: np.ndarray
    T_view_head: np.ndarray


@dataclass
class Measurement:
    # Reference pose: tag frame for a single tag, calibrated wrist frame for bundles.
    T_head_marker: np.ndarray
    reprojection_px: float
    mode: str
    marker_ids: tuple = ()


def projections(pose, points, observation):
    t = observation.T_view_head @ pose
    xyz = points @ t[:3, :3].T + t[:3, 3]
    if np.any(xyz[:, 2] <= 1e-4):
        return None
    homogeneous = xyz @ observation.K.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def residual(vector, points, observations):
    pose = transform(vector[:3], vector[3:])
    arrays = []
    for index, observation in enumerate(observations):
        object_points = points[index] if np.ndim(points) == 3 else points
        projected = projections(pose, object_points, observation)
        if projected is None:
            return None
        arrays.append((projected - observation.pixels).ravel())
    return np.concatenate(arrays)


def refine(initial, points, observations):
    """Small damped least-squares solve across BOTH cameras, without SciPy."""
    vector = np.r_[cv2.Rodrigues(initial[:3, :3])[0].ravel(), initial[:3, 3]]
    error = residual(vector, points, observations)
    if error is None:
        return initial
    damping = 1e-3
    for _ in range(15):
        columns = []
        for index in range(6):
            delta = np.zeros(6); delta[index] = 1e-6
            shifted = residual(vector + delta, points, observations)
            if shifted is None:
                return transform(vector[:3], vector[3:])
            columns.append((shifted - error) / 1e-6)
        jacobian = np.column_stack(columns)
        normal = jacobian.T @ jacobian
        try:
            step = np.linalg.solve(normal + damping*np.diag(np.maximum(np.diag(normal), 1)), -jacobian.T @ error)
        except np.linalg.LinAlgError:
            break
        candidate = residual(vector + step, points, observations)
        if candidate is not None and candidate @ candidate < error @ error:
            vector += step
            error = candidate
            damping = max(1e-8, damping / 3)
            if np.linalg.norm(step) < 1e-7:
                break
        else:
            damping *= 10
    return transform(vector[:3], vector[3:])


def solve(size, observations, quality):
    if not observations:
        return None, "not_seen"
    if len({o.camera for o in observations}) != len(observations):
        return None, "duplicate_camera"
    if quality["require_stereo"] and len(observations) < 2:
        return None, "need_both_cameras"
    points = corners(size)
    for observation in observations:
        pixels = np.asarray(observation.pixels)
        if pixels.shape != (4, 2) or not np.isfinite(pixels).all():
            return None, "invalid_corners"
        if np.min(np.linalg.norm(pixels - np.roll(pixels, 1, axis=0), axis=1)) < quality["min_side_px"]:
            return None, "marker_too_small"
    candidates = []
    for observation in observations:
        try:
            result = cv2.solvePnPGeneric(points, observation.pixels, observation.K, None,
                                         flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            continue
        if not result[0]:
            continue
        for rvec, tvec in zip(result[1], result[2]):
            initial = np.linalg.inv(observation.T_view_head) @ transform(rvec, tvec)
            if not np.isfinite(initial).all():
                continue
            pose = refine(initial, points, observations)
            distance = np.linalg.norm(pose[:3, 3])
            if not quality["min_distance_m"] <= distance <= quality["max_distance_m"]:
                continue
            errors = []
            for obs in observations:
                t = obs.T_view_head @ pose
                # The printed face is +Z in marker coordinates and must face the camera.
                projected = projections(pose, points, obs)
                if projected is None or t[:3, 2] @ t[:3, 3] >= 0:
                    break
                errors.extend(np.linalg.norm(projected - obs.pixels, axis=1))
            else:
                if max(errors) <= quality["max_reprojection_px"]:
                    candidates.append((float(np.sqrt(np.mean(np.square(errors)))), pose))
    if not candidates:
        return None, "inconsistent_geometry"
    candidates.sort(key=lambda item: item[0])
    best_error, best = candidates[0]
    for error, candidate in candidates[1:]:
        different = np.rad2deg(rotation_distance(best[:3, :3], candidate[:3, :3])) > quality["ambiguity_angle_deg"]
        if different and error - best_error < quality["ambiguity_px"]:
            return None, "ambiguous_pose"
    return Measurement(best, best_error, "stereo" if len(observations) == 2 else "mono"), "ok"
