"""Camera optical poses in the board frame, separate from overlay predictions."""
import math

import cv2
import numpy as np


VIEWER_DEFAULTS = {
    "pose_hold_s": .3,
    "pose_row_share": .2,
    "pose_position_decimals": 3,
    "pose_angle_decimals": 1,
}
POSE_METADATA = {
    "position_units": "metres", "quaternion_order": "xyzw", "rpy_units": "degrees",
    "rpy_convention": "R = Rz(yaw) Ry(pitch) Rx(roll); singular yaw=0",
    "held_values": "camera_pose.held_cameras is last valid data for display only, never a current measurement",
}


def configure_pose_viewer(viewer):
    for key, value in VIEWER_DEFAULTS.items():
        viewer.setdefault(key, value)
    hold, share = viewer["pose_hold_s"], viewer["pose_row_share"]
    if not math.isfinite(hold) or hold < 0:
        raise ValueError("viewer.pose_hold_s must be finite and nonnegative")
    if not math.isfinite(share) or not 0 < share < 1:
        raise ValueError("viewer.pose_row_share must be between 0 and 1")
    for key, maximum in (("pose_position_decimals", 6), ("pose_angle_decimals", 3)):
        value = viewer[key]
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f"viewer.{key} must be an integer 0..{maximum}")
    return viewer


def camera_in_board(T_camera_board):
    """Invert the full rigid transform, including the rotated translation."""
    T_board_camera = np.eye(4)
    rotation = T_camera_board[:3, :3].T
    T_board_camera[:3, :3] = rotation
    T_board_camera[:3, 3] = -rotation @ T_camera_board[:3, 3]

    # R = Rz(yaw) Ry(pitch) Rx(roll). At gimbal lock choose yaw=0;
    # retain the quaternion/matrix as the unambiguous orientation.
    horizontal = np.hypot(rotation[0, 0], rotation[1, 0])
    singular = bool(horizontal < 1e-7)
    pitch = np.arctan2(-rotation[2, 0], horizontal)
    if singular:
        roll, yaw = np.arctan2(-rotation[1, 2], rotation[1, 1]), 0.
    else:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    vector = cv2.Rodrigues(rotation)[0].ravel()
    angle = np.linalg.norm(vector)
    xyz = vector * (np.sin(angle/2)/angle if angle > 1e-10 else .5)
    quaternion = np.r_[xyz, np.cos(angle/2)]
    return {"T_board_camera": T_board_camera.tolist(),
            "position_m": T_board_camera[:3, 3].tolist(),
            "quaternion_xyzw": quaternion.tolist(),
            "rpy_deg": np.rad2deg([roll, pitch, yaw]).tolist(),
            "rpy_singular": singular}


def camera_pose_frame(stamp, track, transforms, hold_s=.3):
    """Only accepted measurements populate cameras; held values are explicit.

    transforms maps each camera name to T_camera_reference. The existing
    track's last accepted pose/time defines the hold deadline; neither a
    failed frame nor an overlay prediction can refresh it.
    """
    measurement = track.measurement
    last_stamp = track.last_stamp
    # Do not carry a future measurement across a rewind or another recording.
    if last_stamp is not None and stamp < last_stamp:
        last_stamp = None
        measurement = None
    valid = measurement is not None
    measurement_time = float(stamp) if valid else last_stamp
    age = float(stamp-measurement_time) if measurement_time is not None else None
    held = not valid and age is not None and age <= hold_s and track.last_pose is not None
    state = "valid" if valid else "stale" if held else "lost" if age is not None else "not_seen"
    pose = measurement.pose if valid else track.last_pose if held else None
    derived = {name: camera_in_board(transform @ pose) if pose is not None else None
               for name, transform in transforms.items()}
    empty = {name: None for name in transforms}
    return {"valid": valid, "status": state,
            "measurement_time_s": measurement_time, "age_s": age,
            "observation_mode": measurement.mode if valid else None,
            "reprojection_px": measurement.reprojection_px if valid else None,
            "inliers_by_camera": {name: sum(key[0] == name for key in measurement.inliers)
                                  if valid else 0 for name in transforms},
            "cameras": derived if valid else empty,
            "held_cameras": derived if held else empty}


def pose_document(frame, name, viewer):
    """Compact Markdown, with held numbers labelled separately from live data."""
    position_digits = viewer.get("pose_position_decimals", VIEWER_DEFAULTS["pose_position_decimals"])
    angle_digits = viewer.get("pose_angle_decimals", VIEWER_DEFAULTS["pose_angle_decimals"])
    status = {"valid": "VALID - measured", "stale": "STALE - LAST VALID ONLY",
              "lost": "LOST - board pose unavailable", "not_seen": "WAITING - no board pose"}[frame["status"]]
    age = "--" if frame["age_s"] is None else f"{frame['age_s']:.3f} s"
    value = frame["cameras"][name] or frame["held_cameras"][name]
    def numbers(key, digits):
        if value is None:
            return ["—"]*3
        # Round first to avoid distracting negative zero in the readout.
        return [f"{round(v, digits) or 0.:.{digits}f}" for v in value[key]]
    x, y, z = numbers("position_m", position_digits)
    roll, pitch, yaw = numbers("rpy_deg", angle_digits)
    # Bold live values stay bright; the code style visually mutes held values.
    # TextDocument does not support arbitrary per-span foreground colors.
    def emphasize(value):
        return f"**{value}**" if frame["valid"] else f"`{value}`"
    x, y, z, roll, pitch, yaw = map(emphasize, (x, y, z, roll, pitch, yaw))
    lines = [f"**{status}** | Measurement age: {age}", "",
             f"**Position (m)**   X: {x}   Y: {y}   Z: {z}", "",
             f"**Attitude (deg)**   Roll: {roll}   Pitch: {pitch}   Yaw: {yaw}", ""]
    if frame["valid"]:
        counts = ", ".join(f"{cam}: {count}" for cam, count in frame["inliers_by_camera"].items())
        source = " | via stereo extrinsics" if frame["inliers_by_camera"][name] == 0 else ""
        lines.append(f"{frame['observation_mode'].upper()}{source} | Inlier tags ({counts}) | Joint RMS: {frame['reprojection_px']:.2f} px")
    else:
        lines.append("Last accepted measurement; not the current pose." if value else "No current measurement. Values resume when the board pose is valid.")
    if value and value["rpy_singular"]:
        lines.extend(["", "Euler singularity: yaw set to 0; use the recorded quaternion."])
    return "\n".join(lines)
