"""SO(3) smoothing and bounded trajectories with explicit gaps, never extrapolation."""
from collections import deque
import cv2
import numpy as np
from .pose import rotation_distance, quaternion_xyzw


class Track:
    def __init__(self, target, settings):
        self.target, self.settings = target, settings
        self.offset = np.asarray(target["T_wrist_endpoint"] if "ids" in target else target["T_marker_endpoint"], np.float64)
        self.path = deque(maxlen=settings["trajectory_points"])
        self.history = deque(maxlen=settings["pose_history_count"])
        self.last_time = None
        self.last_raw = None
        self.pose = None
        self.raw_pose = None
        self.quaternion = None
        self.raw_quaternion = None
        self.measurement = None
        self.status = "not_seen"
        self.segment = -1
        self.broken = True
        self.last_history = -float("inf")
        self.last_update = -float("inf")

    def update(self, stamp, measurement, reason):
        if not np.isfinite(stamp) or stamp <= self.last_update:
            raise ValueError("Tracking timestamps must increase")
        self.last_update = stamp
        dt = stamp - self.last_time if self.last_time is not None else float("inf")
        raw = measurement.T_head_marker @ self.offset if measurement is not None else None
        if raw is not None and dt <= self.settings["max_gap_s"]:
            distance = np.linalg.norm(raw[:3, 3] - self.last_raw[:3, 3])
            angle = np.rad2deg(rotation_distance(self.last_raw[:3, :3], raw[:3, :3]))
            if (distance > self.settings["max_speed_m_s"]*dt + self.settings["jump_slack_m"] or
                    angle > self.settings["max_angular_speed_deg_s"]*dt + self.settings["jump_slack_deg"]):
                raw, reason = None, "motion_jump"
        if raw is None:
            self.pose = self.raw_pose = self.quaternion = self.raw_quaternion = self.measurement = None
            self.status, self.broken = reason, True
            return
        if self.broken or dt > self.settings["max_gap_s"]:
            self.segment += 1
            filtered = raw.copy()
            self.last_history = -float("inf")
        else:
            tau = self.settings["smoothing_ms"] / 1000
            alpha = 1.0 if tau == 0 else 1 - np.exp(-dt / tau)
            filtered = np.eye(4)
            filtered[:3, 3] = (1-alpha)*self.pose[:3, 3] + alpha*raw[:3, 3]
            delta = cv2.Rodrigues(self.pose[:3, :3].T @ raw[:3, :3])[0]
            filtered[:3, :3] = self.pose[:3, :3] @ cv2.Rodrigues(alpha*delta)[0]
        q, raw_q = quaternion_xyzw(filtered[:3, :3]), quaternion_xyzw(raw[:3, :3])
        if self.quaternion is not None and np.dot(q, self.quaternion) < 0:
            q = -q
        if self.raw_quaternion is not None and np.dot(raw_q, self.raw_quaternion) < 0:
            raw_q = -raw_q
        self.pose, self.raw_pose = filtered, raw
        self.quaternion, self.raw_quaternion = q, raw_q
        self.last_raw, self.last_time = raw, stamp
        self.measurement, self.status, self.broken = measurement, measurement.mode, False
        self.path.append((self.segment, filtered[:3, 3].copy()))
        if stamp - self.last_history >= self.settings["pose_history_interval_s"]:
            self.history.append(filtered.copy())
            self.last_history = stamp

    def strips(self):
        groups, current, previous = [], [], None
        for segment, point in self.path:
            if segment != previous and current:
                if len(current) >= 2:
                    groups.append(np.asarray(current))
                current = []
            current.append(point)
            previous = segment
        if len(current) >= 2:
            groups.append(np.asarray(current))
        return groups
