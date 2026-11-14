"""Detect tags in undistorted perspective views while preserving raw RGB output."""
import cv2
import numpy as np
from ..common.geometry import rotation
from .config import dictionary
from .pose import Observation, solve, corners, refine, projections
from .bundle import marker_ids, fuse


class Detector:
    def __init__(self, cfg, cameras):
        self.cfg = cfg
        self.last_seen = []
        settings = cfg["detection_view"]
        width, height = settings["width"], settings["height"]
        focal = width / (2*np.tan(np.deg2rad(settings["hfov"])/2))
        self.K = np.array([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1]], np.float64)
        camera_from_view = rotation(settings["yaw"], settings["pitch"])
        view_from_camera = np.eye(4)
        view_from_camera[:3, :3] = camera_from_view.T
        params = cv2.aruco.DetectorParameters()
        params.markerBorderBits = cfg["border_bits"]
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 3
        self.detector = cv2.aruco.ArucoDetector(dictionary(cfg["dictionary"]), params)
        self.views = []
        root_from_head = np.linalg.inv(cameras[cfg["reference"]].T_from_root)
        for channel in cfg["channels"]:
            cam = cameras[channel["camera"]]
            mx, my = cv2.fisheye.initUndistortRectifyMap(cam.K, cam.D, camera_from_view.T,
                                                        self.K, (width, height), cv2.CV_32FC1)
            raw_mask = np.ones((cam.size[1], cam.size[0]), np.uint8)
            rows = int(channel.get("invalid_bottom_rows", 0))
            if rows:
                raw_mask[-rows:] = 0
            if channel.get("valid_mask"):
                mask = cv2.imread(channel["valid_mask"], cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != raw_mask.shape:
                    raise ValueError("Invalid camera validity mask")
                raw_mask &= mask > 0
            valid = cv2.remap(raw_mask.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT) >= 0.999
            t = view_from_camera @ cam.T_from_root @ root_from_head
            self.views.append((cam, mx, my, valid, t))

    def observe(self, images):
        if len(images) != len(self.views):
            raise ValueError("Expected a complete stereo pair")
        found = {i: [] for target in self.cfg["targets"] for i in marker_ids(target)}
        self.last_seen = []
        rejected = set()
        for image, (cam, mx, my, valid, t) in zip(images, self.views):
            if (image.shape[1], image.shape[0]) != cam.size:
                raise ValueError(f"{cam.name} requires native calibrated resolution {cam.size}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            view = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            view[~valid] = 128
            quads, ids, _ = self.detector.detectMarkers(view)
            if ids is None:
                continue
            ids = ids.ravel().tolist()
            for marker_id, quad in zip(ids, quads):
                self.last_seen.append({"camera": cam.name, "id": int(marker_id),
                                       "side_px": round(float(np.mean(np.linalg.norm(
                                           quad.reshape(4, 2) - np.roll(quad.reshape(4, 2), 1, axis=0), axis=1))), 1)})
                if marker_id not in found:
                    continue
                if ids.count(marker_id) > 1:
                    rejected.add(marker_id)
                    continue
                pixels = np.asarray(quad, np.float64).reshape(4, 2)
                xy = pixels.astype(np.float32)
                good = cv2.remap(valid.astype(np.float32), xy[:, 0:1], xy[:, 1:2], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT).ravel() >= 0.999
                if good.all():
                    found[marker_id].append(Observation(cam.name, pixels, self.K, t))
        for marker_id in rejected:
            found[marker_id] = []
        return found, rejected

    def measure(self, images):
        observations, duplicates = self.observe(images)
        measurements = {}
        for target in self.cfg["targets"]:
            for marker_id in marker_ids(target):
                measurement, reason = ((None, "duplicate_marker_id") if marker_id in duplicates else
                                       solve(target["size_m"], observations[marker_id], self.cfg["quality"]))
                if measurement is not None:
                    measurement.marker_ids = (marker_id,)
                measurements[marker_id] = measurement, reason
        self.last_observations = observations
        return measurements

    def combine(self, measurements):
        results = {}
        for target in self.cfg["targets"]:
            if "ids" not in target:
                results[target["name"]] = measurements[target["id"]]
                continue
            if "T_wrist_marker" not in target:
                results[target["name"]] = None, "wrist_calibration_required"
                continue
            measurement, reason = fuse(target, measurements, self.cfg["bundle_fusion"])
            if measurement is not None:
                # Refine a common wrist pose against all accepted tag corners.
                points, observations = [], []
                for marker_id in measurement.marker_ids:
                    matrix = np.asarray(target["T_wrist_marker"][str(marker_id)])
                    xyz = corners(target["size_m"]) @ matrix[:3, :3].T + matrix[:3, 3]
                    for observation in self.last_observations[marker_id]:
                        points.append(xyz)
                        observations.append(observation)
                pose = refine(measurement.T_head_marker, np.asarray(points), observations)
                errors = []
                for xyz, observation in zip(points, observations):
                    projected = projections(pose, xyz, observation)
                    if projected is None:
                        errors = [float("inf")]
                        break
                    errors.extend(np.linalg.norm(projected-observation.pixels, axis=1))
                if max(errors) > self.cfg["quality"]["max_reprojection_px"]:
                    measurement, reason = None, "bundle_reprojection_failed"
                else:
                    measurement.T_head_marker = pose
                    measurement.reprojection_px = float(np.sqrt(np.mean(np.square(errors))))
            results[target["name"]] = measurement, reason
        return results

    def process(self, images):
        return self.combine(self.measure(images))
