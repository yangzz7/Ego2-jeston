"""Measured poses and explicitly separate, bounded display predictions."""
import math
import cv2
import numpy as np
from .pose import rotation_distance

TEMPORAL_DEFAULTS = {
    "detection_interval_s": .25,
    "max_flow_gap_s": .25,
    "flow_fb_max_px": 1.2,
    "flow_error_max": 25.,
    "prediction_max_s": .15,
    "appearance_min_contrast": 18.,
    "appearance_max_bit_errors": 4,
    "tag_grace_s": .12,
}


def configure_tracking(cfg):
    for name, value in TEMPORAL_DEFAULTS.items():
        cfg.setdefault(name, min(value, cfg["max_gap_s"]) if name == "prediction_max_s" else value)
        if not math.isfinite(float(cfg[name])) or cfg[name] <= 0:
            raise ValueError(f"tracking.{name} must be finite and positive")
    if cfg["prediction_max_s"] > cfg["max_gap_s"]:
        raise ValueError("prediction_max_s must not exceed max_gap_s")
    if not isinstance(cfg["appearance_max_bit_errors"], int) or cfg["appearance_max_bit_errors"] > 5:
        raise ValueError("appearance_max_bit_errors must be an integer 1..5")
    return cfg


class Track:
    def __init__(self, cfg):
        self.cfg = configure_tracking(dict(cfg))
        self.last_pose = None
        self.last_stamp = None
        self.measurement = None
        self.status = "not_seen"
        self.display_pose = None
        self.display_predicted = False
        self.prediction_age_s = 0.
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def prior(self, stamp):
        if self.last_stamp is None or stamp-self.last_stamp > self.cfg["max_gap_s"]:
            return None
        return self.last_pose

    def predict(self, stamp):
        previous = self.prior(stamp)
        if previous is None:
            return None
        dt = max(0., stamp-self.last_stamp)
        pose = previous.copy()
        pose[:3, 3] += self.velocity*dt
        pose[:3, :3] = cv2.Rodrigues(self.angular_velocity*dt)[0] @ pose[:3, :3]
        return pose

    def update(self, stamp, measurement, reason, image_support=False):
        self.measurement = None
        self.display_pose = None
        self.display_predicted = False
        self.prediction_age_s = 0.
        self.status = reason
        previous = self.prior(stamp)
        if measurement is None:
            if (image_support and previous is not None and
                0 <= stamp-self.last_stamp <= self.cfg["prediction_max_s"] and
                reason not in ("ambiguous_pose", "pose_jump", "distance_out_of_range")):
                self.display_pose = self.predict(stamp)
                self.display_predicted = True
                self.prediction_age_s = stamp-self.last_stamp
            return
        if previous is not None:
            dt = max(0, stamp-self.last_stamp)
            distance = np.linalg.norm(measurement.pose[:3, 3]-previous[:3, 3])
            angle = np.rad2deg(rotation_distance(previous[:3, :3], measurement.pose[:3, :3]))
            if (distance > self.cfg["jump_slack_m"] + dt*self.cfg["max_speed_m_s"] or
                angle > self.cfg["jump_slack_deg"] + dt*self.cfg["max_angular_speed_deg_s"]):
                self.status = "pose_jump"
                return
            if dt > 1e-5:
                velocity = (measurement.pose[:3, 3]-previous[:3, 3])/dt
                angular = cv2.Rodrigues(measurement.pose[:3, :3] @ previous[:3, :3].T)[0].ravel()/dt
                # Prediction is for brief visual continuity, never a new
                # measurement. Bound both velocities even with jump slack.
                for value, maximum in ((velocity, self.cfg["max_speed_m_s"]),
                                       (angular, np.deg2rad(self.cfg["max_angular_speed_deg_s"]))):
                    value *= min(1., maximum/max(np.linalg.norm(value), 1e-12))
                self.velocity = .5*self.velocity+.5*velocity
                self.angular_velocity = .5*self.angular_velocity+.5*angular
        else:
            self.velocity[:] = self.angular_velocity[:] = 0
        self.measurement = measurement
        self.display_pose = measurement.pose
        self.last_pose, self.last_stamp = measurement.pose.copy(), stamp
        self.status = measurement.mode


def display_source(obs, track):
    if track.display_pose is None:
        return None
    if obs.display_source is not None:
        return obs.display_source
    if track.measurement is not None and obs.key in track.measurement.inliers:
        return "tracked" if obs.source == "flow" else "measured"
    return None
