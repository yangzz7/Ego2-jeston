"""Explicit physical board layout and independent runtime configuration."""
from pathlib import Path
import math

import cv2
import numpy as np

from ..common.config import load_profile, read_yaml, resolve_path
from ..common.geometry import View
from ..common.paths import CONFIGS

DEFAULT_CONFIG = CONFIGS / "aprilgrid.yaml"


class Board:
    """Board origin at grid centre; X right, Y up, Z out of printed face.

    Tag frames share these axes, with origin at each black square centre.
    Decoder corner order is separate: printed bit rotation only affects the
    corner correspondences, never the displayed board-aligned tag axes.
    """
    def __init__(self, data):
        self.data = data
        if data.get("target_type") != "aprilgrid":
            raise ValueError("target_type must be aprilgrid")
        self.rows, self.cols = int(data["tagRows"]), int(data["tagCols"])
        self.size, spacing = float(data["tagSize"]), float(data["tagSpacing"])
        if min(self.rows, self.cols) < 1 or not math.isfinite(self.size) or self.size <= 0:
            raise ValueError("Board rows, columns and tagSize must be positive")
        if not math.isfinite(spacing) or spacing < 0:
            raise ValueError("tagSpacing must be finite and nonnegative")
        self.pitch = self.size * (1 + spacing)
        name = data["dictionary"]
        if not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
            raise ValueError(f"Unknown OpenCV dictionary: {name}")
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        self.border = int(data["border_bits"])
        if self.border < 1:
            raise ValueError("border_bits must be positive")
        self.rotation = int(data["marker_rotation_deg"])
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("marker_rotation_deg must be 0, 90, 180 or 270 (clockwise in print)")
        layout = np.asarray(data["ids_by_row"])
        if layout.shape != (self.rows, self.cols) or not np.issubdtype(layout.dtype, np.integer):
            raise ValueError("ids_by_row must contain integer IDs with tagRows x tagCols entries")
        self.ids = tuple(int(i) for i in layout.ravel())
        if len(set(self.ids)) != len(self.ids) or min(self.ids) < 0 or max(self.ids) >= len(self.dictionary.bytesList):
            raise ValueError("Board IDs must be unique and belong to the dictionary")
        self.centres = {int(layout[r, c]): np.array([(c-(self.cols-1)/2)*self.pitch,
                        ((self.rows-1)/2-r)*self.pitch, 0.])
                        for r in range(self.rows) for c in range(self.cols)}
        h = self.size / 2
        canonical = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        rot = cv2.Rodrigues(np.array([0., 0., -np.deg2rad(self.rotation)]))[0]
        self.decoded_corners = canonical @ rot.T
        self.extent = np.array([(self.cols-1)*self.pitch+self.size,
                                (self.rows-1)*self.pitch+self.size])
        self.page_size = np.asarray(data.get("page_size_m", self.extent + self.size), float)
        if self.page_size.shape != (2,) or not np.isfinite(self.page_size).all() or np.any(self.page_size < self.extent):
            raise ValueError("page_size_m must enclose the grid")

    def corners(self, marker_id):
        return self.decoded_corners + self.centres[marker_id]

    def tag_pose(self, board_pose, marker_id):
        pose = board_pose.copy()
        pose[:3, 3] += pose[:3, :3] @ self.centres[marker_id]
        return pose


def load_config(path=DEFAULT_CONFIG, target_override=None, *, sn=None):
    cfg, cameras = load_profile(path, sn=sn)
    target = (Path(target_override).expanduser().resolve() if target_override else
              resolve_path(cfg["_base"], cfg["target"]))
    cfg["target"] = str(target)
    board = Board(read_yaml(target))
    cfg["target_parameters"] = board.data
    views = cfg["detection"]["views"]
    if not views:
        raise ValueError("Provide at least one detection view")
    for view in views:
        View(**view)
    fraction = cfg["detection"].get("corner_window_fraction", .16)
    window = cfg["detection"].get("corner_window_max_px", 32)
    if not 0 < fraction <= .2 or not isinstance(window, int) or not 3 <= window <= 64:
        raise ValueError("corner window fraction must be 0..0.2; max window must be an integer 3..64")
    for section, keys in {
        "detection": ["min_side_px"],
        "quality": ["max_reprojection_px", "min_distance_m", "max_distance_m",
                    "ambiguity_px", "ambiguity_angle_deg"],
        "tracking": ["max_gap_s", "max_speed_m_s", "max_angular_speed_deg_s",
                     "jump_slack_m", "jump_slack_deg"],
        "viewer": ["axis_length_m"],
    }.items():
        for key in keys:
            value = float(cfg[section][key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{key} must be finite and positive")
    q = cfg["quality"]
    if q["min_distance_m"] >= q["max_distance_m"] or not 0 < q["min_inlier_fraction"] <= 1:
        raise ValueError("Invalid distance range or min_inlier_fraction")
    if not 1 <= cfg["viewer"]["jpeg_quality"] <= 100:
        raise ValueError("jpeg_quality must be 1..100")
    from .tracking import configure_tracking
    configure_tracking(cfg["tracking"])
    from .camera_pose import configure_pose_viewer
    configure_pose_viewer(cfg["viewer"])
    from .stereo_check import configure_stereo_check
    configure_stereo_check(cfg)
    from .display_isp import configure_display_isp
    configure_display_isp(cfg)
    return cfg, cameras, board
