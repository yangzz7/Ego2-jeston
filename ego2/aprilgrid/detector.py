"""Decode overlapping perspective views; retain native fisheye corner pixels."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from ..common.geometry import project, rotation
from .geometry import bearings


@dataclass
class Observation:
    camera: str
    marker_id: int
    pixels: np.ndarray
    outline: np.ndarray
    view_pixels: np.ndarray
    K: np.ndarray
    T_view_ref: np.ndarray
    view_index: int
    reason: str = "ok"
    refinement_shift_px: float = 0.
    refinement_status: str = "not_run"
    pose_reason: str = "not_solved"
    reprojection_px: float | None = None
    source: str = "decoded"
    flow_error_px: float | None = None
    appearance_score: float | None = None
    appearance_valid: bool = False
    appearance_weak: bool = False
    confirmation_age_s: float = 0.
    display_source: str | None = None
    display_pixels: np.ndarray | None = None
    display_outline: np.ndarray | None = None

    @property
    def key(self):
        return self.camera, self.marker_id


class Detector:
    def __init__(self, cfg, cameras, board):
        self.cfg, self.board, self.cameras = cfg, board, cameras
        self.views = {}
        self.native_masks = {}
        self.remap_tables = {}
        ref_inverse = np.linalg.inv(cameras[cfg["reference"]].T_from_root)
        for ch in cfg["channels"]:
            cam = cameras[ch["camera"]]
            mask = np.ones(cam.size[::-1], np.float32)
            rows = int(ch.get("invalid_bottom_rows", 0))
            if not 0 <= rows < cam.size[1]:
                raise ValueError("invalid_bottom_rows outside image")
            if rows:
                mask[-rows:] = 0
            if ch.get("valid_mask"):
                extra = cv2.imread(ch["valid_mask"], cv2.IMREAD_GRAYSCALE)
                if extra is None or extra.shape != mask.shape:
                    raise ValueError("Invalid camera validity mask")
                mask *= extra > 0
            self.native_masks[cam.name] = mask
            settings = cv2.aruco.DetectorParameters()
            settings.markerBorderBits = board.border
            settings.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            settings.cornerRefinementWinSize = 3
            decoder = cv2.aruco.ArucoDetector(board.dictionary, settings)
            views = []
            for entry in cfg["detection"]["views"]:
                w, h = entry["width"], entry["height"]
                f = w / (2*np.tan(np.deg2rad(entry["hfov"])/2))
                k = np.array([[f, 0, (w-1)/2], [0, f, (h-1)/2], [0, 0, 1.]])
                r = rotation(entry["yaw"], entry["pitch"])
                yy, xx = np.mgrid[:h, :w]
                rays = np.stack([(xx-k[0, 2])/f, (yy-k[1, 2])/f, np.ones_like(xx)], axis=-1) @ r.T
                u, v, ok = project(cam, rays)
                mx, my = u.astype(np.float32), v.astype(np.float32)
                ok &= cv2.remap(mask, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) >= .999
                mx[~ok], my[~ok] = -1, -1
                t = np.eye(4)
                t[:3, :3] = r.T
                t = t @ cam.T_from_root @ ref_inverse
                views.append((mx, my, ok.astype(np.float32), k, r, t))
            self.views[cam.name] = decoder, views
            # OpenCV INTER_LINEAR uses these same tabulated coordinates for
            # float maps. Convert once, rather than on every full-view remap.
            self.remap_tables[cam.name] = [cv2.convertMaps(v[0], v[1], cv2.CV_16SC2) for v in views]
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aprilgrid-detect")

    @staticmethod
    def to_raw(cam, pixels, k, rotation_matrix):
        rays = np.c_[(pixels[:, 0]-k[0, 2])/k[0, 0],
                     (pixels[:, 1]-k[1, 2])/k[1, 1], np.ones(len(pixels))]
        u, v, valid = project(cam, rays @ rotation_matrix.T)
        return np.c_[u, v], valid

    def _eye(self, name, image):
        cam = self.cameras[name]
        if (image.shape[1], image.shape[0]) != cam.size:
            raise ValueError(f"{name} requires calibrated resolution {cam.size}")
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        decoder, views = self.views[name]
        found, scores, duplicates = {}, {}, set()
        for index, (mx, my, valid, k, r, t) in enumerate(views):
            map1, map2 = self.remap_tables[name][index]
            view = cv2.remap(gray, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=128)
            quads, ids, _ = decoder.detectMarkers(view)
            if ids is None:
                continue
            for marker_id, quad in zip(ids.ravel(), quads):
                marker_id = int(marker_id)
                if marker_id not in self.board.centres:
                    continue
                pixels = quad.reshape(4, 2).astype(float)
                good = cv2.remap(valid, pixels[:, 0:1].astype(np.float32),
                                 pixels[:, 1:2].astype(np.float32), cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT).ravel() >= .999
                raw, ok = self.to_raw(cam, pixels, k, r)
                if not good.all() or not ok.all():
                    continue
                side = np.min(np.linalg.norm(raw-np.roll(raw, 1, axis=0), axis=1))
                obs = Observation(name, marker_id, raw, raw, pixels, k, t, index,
                                  "ok" if side >= self.cfg["detection"]["min_side_px"] else "too_small")
                # A large quad near a virtual image edge can be mostly stretch.
                # Prefer native sampling and an optical axis close to the tag.
                angular = np.max(np.linalg.norm((pixels-k[:2, 2])/np.diag(k)[:2], axis=1))
                score = min(side, np.min(np.linalg.norm(pixels-np.roll(pixels, 1, axis=0), axis=1)))/(1+angular*angular)
                if marker_id in found:
                    old = found[marker_id]
                    # A spatially distinct copy of an ID is not another virtual-view detection.
                    if np.linalg.norm(old.pixels.mean(axis=0)-raw.mean(axis=0)) > max(6, side*.5):
                        duplicates.add(marker_id)
                    if score <= scores[marker_id]:
                        continue
                found[marker_id], scores[marker_id] = obs, score
        for marker_id in duplicates:
            found[marker_id].reason = "duplicate_id"
        self.refine_corners(cam, gray, found.values())
        return list(found.values())

    def refine_corners(self, cam, gray, observations):
        """Refine once in the original image, with windows scaled to each corner.

        The PDF's diagonal black squares make a small-window contour corner
        systematically inset. A larger local saddle window reaches the actual
        intersection; limiting it to the border avoids the inner code cells.
        """
        fraction = min(self.cfg["detection"].get("corner_window_fraction", .16),
                       .8*self.board.border/(self.board.dictionary.markerSize+2*self.board.border))
        maximum = self.cfg["detection"].get("corner_window_max_px", 32)
        observations = list(observations)
        if not observations:
            return
        raw = np.array([o.pixels for o in observations])
        edges = np.linalg.norm(raw-np.roll(raw, -1, axis=1), axis=2)
        windows = np.clip(np.rint(np.minimum(edges, np.roll(edges, 1, axis=1))*fraction), 3, maximum).astype(int)
        refined = raw.astype(np.float32).reshape(-1, 2).copy()
        usable = np.repeat([o.reason == "ok" for o in observations], 4)
        for window in np.unique(windows):
            indices = np.flatnonzero((windows.ravel() == window) & usable)
            if not len(indices):
                continue
            points = refined[indices].reshape(-1, 1, 2).copy()
            cv2.cornerSubPix(gray, points, (int(window), int(window)), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            refined[indices] = points[:, 0]
        refined = refined.reshape(-1, 4, 2)
        for i, obs in enumerate(observations):
            corners = refined[i]
            shift = np.linalg.norm(corners-obs.pixels, axis=1)
            valid = (np.isfinite(corners).all() and np.all(shift <= windows[i]) and
                     cv2.isContourConvex(corners.reshape(-1, 1, 2)) and
                     np.all(corners >= 0) and np.all(corners < np.array(cam.size)-1))
            if valid:
                valid = np.all(cv2.remap(self.native_masks[cam.name], corners[:, :1], corners[:, 1:],
                                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) >= .999)
            if not valid:
                obs.refinement_status = "unsafe_shift"
                refined[i] = obs.pixels
            else:
                obs.refinement_shift_px = float(shift.max())
                obs.refinement_status = "refined" if obs.reason == "ok" else "not_run"
        all_rays = bearings(cam, refined.reshape(-1, 2)).reshape(-1, 4, 3)
        for obs, corners, rays in zip(observations, refined, all_rays):
            # Keep the original-view correspondences consistent for PnP and
            # recompute curved outlines from the refined corners.
            r = self.views[cam.name][1][obs.view_index][4]
            rays = rays @ r
            if np.any(rays[:, 2] <= .01):
                obs.refinement_status = "outside_view"
                pixels = obs.view_pixels
            else:
                pixels = rays @ obs.K.T
                pixels = pixels[:, :2]/pixels[:, 2:]
                obs.pixels, obs.view_pixels = corners.astype(float), pixels
            edge = np.concatenate([a + np.linspace(0, 1, 9)[:, None]*(b-a)
                                   for a, b in zip(pixels, np.roll(pixels, -1, axis=0))])
            obs.outline, _ = self.to_raw(cam, edge, obs.K, r)

    def observe(self, images):
        if len(images) != 2:
            raise ValueError("A complete stereo pair is required")
        jobs = [self.pool.submit(self._eye, ch["camera"], image)
                for ch, image in zip(self.cfg["channels"], images)]
        return [obs for job in jobs for obs in job.result()]

    def close(self):
        self.pool.shutdown(wait=True)
