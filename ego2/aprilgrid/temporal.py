"""Image-supported tag tracking with periodic decoding and rigid-board recovery.

Only decoded or optical-flow corners enter the pose solver. Model-projected
corners can supply a visible overlay after a known-ID appearance check, but
never feed back into estimation or seed optical flow.
"""
from dataclasses import replace
import time

import cv2
import numpy as np

from ..common.geometry import project
from .detector import Detector, Observation
from .geometry import bearings
from .pose import Solver
from .tracking import Track, configure_tracking


class Pipeline:
    def __init__(self, cfg, cameras, board, *, throttle_empty=False):
        self.cfg, self.cameras, self.board = cfg, cameras, board
        self.throttle_empty = throttle_empty
        self.settings = configure_tracking(cfg["tracking"])
        self.detector = Detector(cfg, cameras, board)
        self.solver = Solver(cfg, cameras, board)
        self.track = Track(self.settings)
        self.previous_gray, self.previous_obs = {}, {}
        self.last_stamp = self.last_decode = None
        self.known_ids = set()
        self.last_verified = {}
        self.detection_ms = self.solve_ms = 0.
        self.decoded_observations = []
        cells = board.dictionary.markerSize+2*board.border
        yy, xx = np.mgrid[:cells, :cells]
        centres = np.stack([xx+.5, yy+.5], axis=-1).reshape(-1, 2)
        offsets = np.array([[-.15, -.15], [.15, -.15], [-.15, .15], [.15, .15]])
        self.samples = ((centres[:, None]+offsets)/cells).reshape(-1, 1, 2).astype(np.float32)
        self.square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        self.inner = ((xx >= board.border) & (xx < cells-board.border) &
                      (yy >= board.border) & (yy < cells-board.border)).ravel()
        self.patterns = {i: (cv2.aruco.generateImageMarker(board.dictionary, i, cells,
                         borderBits=board.border) > 0).ravel() for i in board.ids}

    def close(self):
        self.detector.close()

    def _select_views(self, name, observations):
        if not observations:
            return
        rays = bearings(self.cameras[name], np.concatenate([o.pixels for o in observations])).reshape(-1, 4, 3)
        views = self.detector.views[name][1]
        local_views = np.stack([rays @ view[4] for view in views])
        indices = np.argmax(local_views[..., 2].min(axis=2), axis=0)
        for row, (obs, index) in enumerate(zip(observations, indices)):
            _, _, _, k, r, t = views[index]
            local = local_views[index, row]
            uv = local @ k.T
            obs.view_index, obs.K, obs.T_view_ref = index, k, t
            obs.view_pixels = uv[:, :2]/uv[:, 2:]

    def _appearances(self, observations, gray):
        from .appearance import verify
        verify(self, observations, gray)

    def _appearance(self, obs, gray):
        self._appearances([obs], gray)
        return obs.appearance_valid

    def _flow_eye(self, name, gray):
        previous = self.previous_obs.get(name, [])
        if not previous:
            return []
        points = np.concatenate([o.pixels for o in previous]).astype(np.float32).reshape(-1, 1, 2)
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, .01))
        moved, forward, error = cv2.calcOpticalFlowPyrLK(self.previous_gray[name], gray, points, None, **options)
        if moved is None:
            return []
        finite = np.isfinite(moved).all(axis=(1, 2))
        safe_moved = moved.copy()
        safe_moved[~finite] = points[~finite]
        back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray[name], safe_moved, None, **options)
        if back is None:
            return []
        fb = np.linalg.norm(back-points, axis=2).ravel()
        good = ((forward.ravel() != 0) & (backward.ravel() != 0) &
                (fb <= self.settings["flow_fb_max_px"]) &
                (error.ravel() <= self.settings["flow_error_max"]) & finite)
        moved = moved.reshape(-1, 4, 2)
        found = []
        cam = self.cameras[name]
        for i, old in enumerate(previous):
            corners = moved[i]
            if (not good[i*4:i*4+4].all() or not cv2.isContourConvex(corners) or
                np.any(corners < 1) or np.any(corners >= np.array(cam.size)-2)):
                continue
            area = cv2.contourArea(corners, oriented=True)
            old_area = cv2.contourArea(old.pixels.astype(np.float32), oriented=True)
            if abs(old_area) < 1 or not .5 <= area/old_area <= 2:
                continue
            if np.min(np.linalg.norm(corners-np.roll(corners, 1, axis=0), axis=1)) < self.cfg["detection"]["min_side_px"]:
                continue
            found.append(replace(old, pixels=corners.astype(float), source="flow", reason="ok",
                         pose_reason="not_solved", reprojection_px=None, display_source=None,
                         display_pixels=None, display_outline=None,
                         confirmation_age_s=0.,
                         appearance_score=None, flow_error_px=float(fb[i*4:i*4+4].max())))
        self._select_views(name, found)
        self.detector.refine_corners(cam, gray, found)
        self._appearances(found, gray)
        return [obs for obs in found if obs.appearance_valid]

    def _merge(self, flow, decoded, gray):
        merged = {o.key: o for o in flow}
        for name in gray:
            self._appearances([o for o in decoded if o.camera == name and o.reason == "ok"], gray[name])
        for obs in decoded:
            old = merged.get(obs.key)
            # Fresh decoding confirms identity. Keep stable, image-verified
            # tracked corners when the two detections refer to the same quad.
            if (old is not None and obs.reason == "ok" and
                np.max(np.linalg.norm(old.pixels-obs.pixels, axis=1)) <= 4):
                continue
            merged[obs.key] = obs
        return list(merged.values())

    def _model_observations(self, pose, gray, keys, stamp):
        result = []
        for name, cam in self.cameras.items():
            ids = sorted(i for n, i in keys if n == name)
            if not ids:
                continue
            t = self.solver.transforms[name] @ pose
            if t[:3, 2] @ t[:3, 3] >= 0:
                continue
            points = np.concatenate([self.board.corners(i) for i in ids])
            xyz = points @ t[:3, :3].T+t[:3, 3]
            u, v, valid = project(cam, xyz)
            raw = np.c_[u, v].reshape(-1, 4, 2)
            visible = []
            usable = (valid.reshape(-1, 4).all(axis=1) & (raw >= 1).all(axis=(1, 2)) &
                      (raw < np.array(cam.size)-2).all(axis=(1, 2)) &
                      (np.linalg.norm(raw-np.roll(raw, 1, axis=1), axis=2).min(axis=1) >=
                       self.cfg['detection']['min_side_px']))
            for marker_id, pixels, good in zip(ids, raw, usable):
                if not good:
                    continue
                visible.append(Observation(name, marker_id, pixels, pixels, pixels, cam.K,
                               self.solver.transforms[name], 0, source="board"))
            self._select_views(name, visible)
            self._appearances(visible, gray[name])
            accepted = []
            for obs in visible:
                if not obs.appearance_valid:
                    age = stamp-self.last_verified.get(obs.key, float("-inf"))
                    if (self.track.display_predicted or not obs.appearance_weak or
                        not 0 <= age <= self.settings["tag_grace_s"]):
                        continue
                    obs.confirmation_age_s = age
                accepted.append(obs)
            # Project all curved tag borders from one view together.
            for index in {o.view_index for o in accepted}:
                group = [o for o in accepted if o.view_index == index]
                pixels = np.array([o.view_pixels for o in group])
                edge = pixels[:, :, None, :] + np.linspace(0, 1, 9)[None, None, :, None]*(
                    np.roll(pixels, -1, axis=1)-pixels)[:, :, None, :]
                r = self.detector.views[name][1][index][4]
                outline, _ = self.detector.to_raw(cam, edge.reshape(-1, 2), group[0].K, r)
                for obs, points in zip(group, outline.reshape(-1, 36, 2)):
                    obs.outline = points
            result.extend(accepted)
        return result

    def process(self, images, stamp):
        if len(images) != 2:
            raise ValueError("A complete stereo pair is required")
        begin = time.perf_counter()
        names = [ch["camera"] for ch in self.cfg["channels"]]
        gray = {name: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for name, image in zip(names, images)}
        reset = self.last_stamp is not None and stamp <= self.last_stamp
        if reset:
            self.track = Track(self.settings)
            self.last_decode = None
            self.last_verified.clear()
        continuous = self.last_stamp is not None and 0 < stamp-self.last_stamp <= self.settings["max_flow_gap_s"]
        flow = []
        if continuous:
            jobs = [self.detector.pool.submit(self._flow_eye, name, gray[name]) for name in names]
            flow = [obs for job in jobs for obs in job.result()]
        flow_finished = time.perf_counter()
        # Search immediately on first frame, tracking loss or a time reset.
        # With no usable track, bound repeated empty full-image searches to the
        # configured refresh period; intervening frames publish no measured pose.
        lost_track = not flow and (any(self.previous_obs.values()) or self.track.prior(stamp) is not None)
        decode = (self.last_decode is None or lost_track or not continuous and self.last_stamp is not None or
                  not flow and not self.throttle_empty or
                  stamp-self.last_decode >= self.settings["detection_interval_s"])
        decode_started = time.perf_counter()
        decoded = self.detector.observe([gray[name] for name in names]) if decode else []
        self.decode_ms = (time.perf_counter()-decode_started)*1000
        if decode:
            self.last_decode = stamp
        observations = self._merge(flow, decoded, gray)
        detected = time.perf_counter()
        prior = self.track.predict(stamp)
        m, reason = self.solver.solve(observations, prior)
        self.solve_ms = (time.perf_counter()-detected)*1000
        reacquired = False
        if m is None and not decode and flow:
            search_started = time.perf_counter()
            decoded = self.detector.observe([gray[name] for name in names])
            self.decode_ms += (time.perf_counter()-search_started)*1000
            observations = self._merge(flow, decoded, gray)
            self.last_decode, decode = stamp, True
            detected += time.perf_counter()-search_started
            solve_started = time.perf_counter()
            m, reason = self.solver.solve(observations, prior)
            self.solve_ms += (time.perf_counter()-solve_started)*1000
        if m is None and flow and decoded:
            # A stale flow track must not veto a fresh, independently verified
            # solution. Require fresh observations from every currently active
            # eye: contradictory stereo must never silently become monocular.
            active_cameras = {o.camera for o in observations if o.reason == "ok"}
            fresh_cameras = {o.camera for o in decoded if o.reason == "ok"}
            if active_cameras == fresh_cameras:
                previous_diagnostics = self.solver.diagnostics
                fresh = [replace(o, display_source=None) for o in decoded]
                solve_started = time.perf_counter()
                fresh_m, fresh_reason = self.solver.solve(fresh, prior)
                self.solve_ms += (time.perf_counter()-solve_started)*1000
                if fresh_m is not None:
                    m, reason, observations = fresh_m, fresh_reason, fresh
                    reacquired = True
                else:
                    self.solver.diagnostics = previous_diagnostics
        self.detection_ms = (detected-begin)*1000
        self.track.update(stamp, m, reason, image_support=bool(flow))
        if self.track.measurement is not None:
            self.known_ids.update(key[1] for key in m.inliers)
        # Save only actual image measurements. Display projections are never
        # optical-flow seeds, and never become synthetic pose correspondences.
        # A corner can pass local appearance yet disagree with rigid-board
        # geometry. Do not carry such outliers into the next pose fit forever.
        measured = self.track.measurement
        self.previous_obs = {name: [o for o in observations if o.camera == name and o.reason == "ok"
                                   and o.appearance_valid and (measured is None or o.key in measured.inliers)]
                             for name in names}
        self.previous_gray, self.last_stamp = gray, stamp
        recovery_started = time.perf_counter()
        pose = self.track.display_pose
        recovered = 0
        if pose is not None:
            accepted = self.track.measurement.inliers if self.track.measurement is not None else frozenset()
            for obs in observations:
                if obs.key in accepted:
                    obs.display_source = "tracked" if obs.source == "flow" else "measured"
            missing = {(name, i) for name in names for i in self.known_ids} - accepted
            self.last_verified.update({key: stamp for key in accepted})
            missing -= {o.key for o in observations if o.reason != "ok"}
            model = self._model_observations(pose, gray, missing, stamp)
            lookup = {o.key: o for o in observations}
            for projected in model:
                projected.display_source = ("predicted" if self.track.display_predicted else
                                            "board" if projected.appearance_valid else "board_grace")
                if projected.appearance_valid and not self.track.display_predicted:
                    self.last_verified[projected.key] = stamp
                if projected.key in lookup:
                    # Preserve measured corners and rejection diagnostics in the
                    # JSONL; only the overlay is inferred from the rigid board.
                    lookup[projected.key].display_source = projected.display_source
                    lookup[projected.key].display_pixels = projected.pixels
                    lookup[projected.key].display_outline = projected.outline
                    lookup[projected.key].confirmation_age_s = projected.confirmation_age_s
                else:
                    observations.append(projected)
                recovered += 1
        if self.track.display_predicted and recovered == 0:
            self.track.display_pose = None
            self.track.display_predicted = False
            self.track.prediction_age_s = 0.
        self.solver.diagnostics["temporal"] = {
            "decoded_this_frame": decode, "decoded_tags": len(decoded), "flow_tags": len(flow),
            "board_display_tags": recovered, "predicted": self.track.display_predicted,
            "prediction_age_s": self.track.prediction_age_s,
            "known_ids": len(self.known_ids),
            "pose_reacquired": reacquired,
        }
        # Keep this frame's fresh detector corners separate from the merged
        # joint-fit/flow/model overlays. Exit-time stereo validation must not
        # select corners using the supplied stereo extrinsics or its inliers.
        self.decoded_observations = decoded
        self.recovery_ms = (time.perf_counter()-recovery_started)*1000
        self.flow_ms = (flow_finished-begin)*1000
        self.detection_ms = (time.perf_counter()-begin)*1000-self.solve_ms
        return observations
