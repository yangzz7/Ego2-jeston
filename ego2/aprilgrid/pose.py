"""Planar initialization and robust joint optimization in native fisheye pixels."""
from dataclasses import dataclass
from collections import defaultdict
import math

import cv2
import numpy as np

from ..common.geometry import project
from .geometry import bearings, projection_jacobian


def transform(vector):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(vector[:3], float))[0]
    result[:3, 3] = vector[3:]
    return result


def rotation_distance(a, b):
    return float(np.linalg.norm(cv2.Rodrigues(a.T @ b)[0]))


@dataclass
class Measurement:
    pose: np.ndarray  # T_reference_board, metres
    reprojection_px: float
    mode: str
    inliers: frozenset
    errors: dict
    refit_rounds: int = 0


class Solver:
    def __init__(self, cfg, cameras, board):
        self.cfg, self.cameras, self.board = cfg, cameras, board
        ref_inverse = np.linalg.inv(cameras[cfg["reference"]].T_from_root)
        self.transforms = {name: cam.T_from_root @ ref_inverse for name, cam in cameras.items()}
        self.diagnostics = {}

    def pack(self, observations):
        groups = defaultdict(list)
        for index, obs in enumerate(observations):
            groups[obs.camera].append((index, obs))
        return [(name, np.array([i for i, _ in entries]),
                 np.concatenate([self.board.corners(o.marker_id) for _, o in entries]),
                 np.concatenate([o.pixels for _, o in entries])) for name, entries in groups.items()]

    def residual(self, pose, packed, count):
        result = np.empty((count, 4, 2))
        for name, indices, points, pixels in packed:
            t = self.transforms[name] @ pose
            xyz = points @ t[:3, :3].T + t[:3, 3]
            u, v, valid = project(self.cameras[name], xyz)
            if not valid.all() or t[:3, 2] @ t[:3, 3] >= 0:
                return None
            result[indices] = (np.c_[u, v] - pixels).reshape(-1, 4, 2)
        return result.reshape(-1)

    def refine(self, initial, observations, iterations=10):
        prepared = getattr(self, '_refine_inputs', {}).get(tuple(id(o) for o in observations))
        packed = self.pack(observations) if prepared is None else prepared[0]
        vector = np.r_[cv2.Rodrigues(initial[:3, :3])[0].ravel(), initial[:3, 3]]
        error = self.residual(initial, packed, len(observations))
        if error is None:
            return initial
        threshold = self.cfg["quality"]["max_reprojection_px"]
        # Unequal visibility must not let one eye overwhelm the other. A capped
        # density weight keeps a small stereo overlap influential, while all
        # acceptance thresholds remain in unweighted native pixels.
        if prepared is None:
            density = np.ones((len(observations), 4))
            largest = max(len(indices) for _, indices, _, _ in packed)
            for _, indices, _, _ in packed:
                density[indices] = min(2., np.sqrt(largest/len(indices)))
            density = density.ravel()
            if hasattr(self, '_refine_inputs'):
                self._refine_inputs[tuple(id(o) for o in observations)] = packed, density
        else:
            density = prepared[1]

        def cost(e):
            lengths = np.linalg.norm(e.reshape(-1, 2), axis=1)
            return np.sum(density*np.where(lengths < threshold, lengths**2, 2*threshold*lengths-threshold**2))

        damping = 1e-3
        current_cost = cost(error)
        for _ in range(iterations):
            jac = self.jacobian(vector, packed, len(observations))
            weights = np.repeat(density*np.minimum(1, threshold/np.maximum(
                np.linalg.norm(error.reshape(-1, 2), axis=1), 1e-9)), 2)
            normal = jac.T @ (weights[:, None]*jac)
            try:
                step = np.linalg.solve(normal + damping*np.diag(np.maximum(normal.diagonal(), 1)),
                                       -jac.T @ (weights*error))
            except np.linalg.LinAlgError:
                break
            candidate = self.residual(transform(vector+step), packed, len(observations))
            candidate_cost = cost(candidate) if candidate is not None else float('inf')
            if candidate is not None and candidate_cost < current_cost:
                vector += step
                error = candidate
                current_cost = candidate_cost
                damping = max(1e-8, damping/3)
                if np.linalg.norm(step) < 1e-7:
                    break
            else:
                damping *= 10
                if np.linalg.norm(step) < 1e-8:
                    break
        return transform(vector)

    def jacobian(self, vector, packed, count):
        rot, derivative = cv2.Rodrigues(vector[:3])
        derivative = derivative.reshape(3, 3, 3)
        result = np.empty((count, 4, 2, 6))
        for name, indices, points, _ in packed:
            t = self.transforms[name]
            xyz = (points @ rot.T+vector[3:]) @ t[:3, :3].T+t[:3, 3]
            dxyz = np.empty((len(points), 3, 6))
            dxyz[:, :, :3] = np.einsum("ij,njk->nik", t[:3, :3],
                                       np.einsum("nj,akj->nka", points, derivative))
            dxyz[:, :, 3:] = t[:3, :3]
            jac = np.einsum("nij,njk->nik", projection_jacobian(self.cameras[name], xyz), dxyz)
            result[indices] = jac.reshape(-1, 4, 2, 6)
        return result.reshape(-1, 6)

    def initial_poses(self, observations, prior):
        groups = defaultdict(list)
        for obs in observations:
            groups[obs.camera].append(obs)
        seeds = [prior] if prior is not None else []
        # One bearing chart per eye avoids fragmenting a large board between
        # overlapping detection views. It is also independent of corner shifts.
        for name, group in groups.items():
            points = np.concatenate([self.board.corners(o.marker_id) for o in group])
            rays = bearings(self.cameras[name], np.concatenate([o.pixels for o in group]))
            z = rays.mean(axis=0)
            z /= np.linalg.norm(z)
            up = [0., 1., 0.] if abs(z[1]) < .9 else [1., 0., 0.]
            x = np.cross(up, z)
            x /= np.linalg.norm(x)
            rotation = np.column_stack([x, np.cross(z, x), z])
            local = rays @ rotation
            valid = local[:, 2] > .1
            if valid.sum() < 4:
                continue
            points = points[valid]
            pixels = local[valid, :2]/local[valid, 2:]*500
            k = np.diag([500., 500., 1.])
            t = np.eye(4)
            t[:3, :3] = rotation.T
            t = t @ self.transforms[name]
            try:
                subsets = [(points, pixels)]
                if len(group) > 1:
                    _, mask = cv2.findHomography(points[:, :2], pixels, cv2.RANSAC, 3.)
                    if mask is not None and 4 <= mask.sum() < len(points):
                        indices = mask.ravel().astype(bool)
                        subsets.insert(0, (points[indices], pixels[indices]))
                for xyz, uv in subsets:
                    answer = cv2.solvePnPGeneric(np.ascontiguousarray(xyz), np.ascontiguousarray(uv),
                                                 k, None, flags=cv2.SOLVEPNP_IPPE)
                    if not answer[0]:
                        continue
                    for rvec, tvec in zip(answer[1], answer[2]):
                        seed = np.linalg.inv(t) @ transform(np.r_[rvec.ravel(), tvec.ravel()])
                        if np.isfinite(seed).all() and not any(
                            np.linalg.norm(seed[:3, 3]-s[:3, 3]) < .001 and
                            rotation_distance(seed[:3, :3], s[:3, :3]) < .01 for s in seeds):
                            seeds.append(seed)
            except cv2.error:
                continue
        return seeds

    def solve(self, observations, prior=None):
        # The same corner set is refined from several independent IPPE seeds.
        # Reuse its packing/weights only within this solve; never across frames
        # or after callers edit pixels/calibration. All seeds/gates are retained.
        self._refine_inputs = {}
        try:
            return self._solve(observations, prior)
        finally:
            del self._refine_inputs

    def _solve(self, observations, prior=None):
        for obs in observations:
            obs.pose_reason, obs.reprojection_px = "not_solved", None
        observations = [o for o in observations if o.reason == "ok"]
        self.diagnostics = {"observations": len(observations), "inliers": 0, "seeds": 0}
        if not observations:
            self.diagnostics["reason"] = "no_usable_tags"
            return None, "no_usable_tags"
        quality = self.cfg["quality"]
        cameras = {o.camera for o in observations}
        packed = self.pack(observations)
        candidates = []
        required = math.ceil(len(observations)*quality["min_inlier_fraction"])
        self.diagnostics["required_inliers"] = required
        best_failed = None
        seeds = self.initial_poses(observations, prior)
        self.diagnostics["seeds"] = len(seeds)
        refined = []
        for initial in seeds:
            pose = self.refine(initial, observations)
            if any(np.linalg.norm(pose[:3, 3]-p[:3, 3]) < 1e-5 and
                   rotation_distance(pose[:3, :3], p[:3, :3]) < 1e-4 for p in refined):
                continue
            refined.append(pose)
            previous, visited = None, set()
            reason = "invalid_projection"
            rms = mask = lengths = None
            for iteration in range(8):
                error = self.residual(pose, packed, len(observations))
                if error is None:
                    reason = "invalid_projection"
                    break
                lengths = np.linalg.norm(error.reshape(-1, 4, 2), axis=2)
                rms = np.sqrt(np.mean(lengths**2, axis=1))
                mask = (rms <= quality["max_reprojection_px"]) & (lengths.max(axis=1) <= 2*quality["max_reprojection_px"])
                # Reclassify after every refit. If a mask cycles, shrink the
                # current fitting set; never accept stale pre-refit membership.
                if mask.tobytes() in visited and previous is not None:
                    mask &= previous
                inliers = [o for o, good in zip(observations, mask) if good]
                distance = float(np.linalg.norm(pose[:3, 3]))
                # Keep every independently verified consensus. Local refitting
                # may lose support (especially with unequal eye visibility);
                # that must not discard a previous pose that passed all gates.
                if (mask.sum() >= required and {o.camera for o in inliers} == cameras and
                    quality["min_distance_m"] <= distance <= quality["max_distance_m"]):
                    candidates.append(Measurement(pose.copy(), float(np.sqrt(np.mean(lengths[mask]**2))),
                        "stereo" if len(cameras) > 1 else "mono", frozenset(o.key for o in inliers),
                        {o.key: float(e) for o, e in zip(observations, rms)}, iteration))
                if {o.camera for o in inliers} != cameras:
                    reason = "stereo_inconsistent" if len(cameras) > 1 else "insufficient_inliers"
                    break
                if (previous is not None and np.array_equal(mask, previous)) or mask.all():
                    reason = "ok" if mask.sum() >= required else "insufficient_inliers"
                    break
                if len(inliers) < max(2, math.ceil(len(observations)*.25)):
                    reason = "insufficient_inliers"
                    break
                if iteration == 7:
                    reason = "unstable_inliers"
                    break
                visited.add(mask.tobytes())
                previous = mask.copy()
                pose = self.refine(pose, inliers, iterations=6)
            distance = float(np.linalg.norm(pose[:3, 3]))
            if reason == "ok" and not quality["min_distance_m"] <= distance <= quality["max_distance_m"]:
                reason = "distance_out_of_range"
            diagnostic = {"reason": reason, "inliers": int(mask.sum()) if mask is not None else 0,
                          "refit_rounds": iteration, "distance_m": distance}
            if rms is not None:
                diagnostic["inliers_by_camera"] = {name: sum(bool(good) for o, good in zip(observations, mask)
                                                           if o.camera == name) for name in cameras}
                diagnostic["median_error_by_camera_px"] = {name: float(np.median([
                    error for o, error in zip(observations, rms) if o.camera == name])) for name in cameras}
                diagnostic["median_tag_error_px"] = float(np.median(rms))
                key = (-diagnostic["inliers"], diagnostic["median_tag_error_px"])
                if best_failed is None or key < best_failed[0]:
                    best_failed = key, diagnostic, rms, mask
        if not candidates:
            reason = "initialization_failed" if not seeds else "invalid_projection"
            if best_failed is not None:
                _, diagnostic, rms, mask = best_failed
                self.diagnostics.update(diagnostic)
                reason = diagnostic["reason"]
                for obs, error, good in zip(observations, rms, mask):
                    obs.reprojection_px = float(error)
                    obs.pose_reason = reason if good else "reprojection_outlier"
            else:
                for obs in observations:
                    obs.pose_reason = reason
            self.diagnostics["reason"] = reason
            return None, reason
        candidates.sort(key=lambda m: (-len(m.inliers), m.reprojection_px))
        best = candidates[0]
        self.diagnostics.update(inliers=len(best.inliers), reprojection_px=best.reprojection_px,
            refit_rounds=best.refit_rounds, distance_m=float(np.linalg.norm(best.pose[:3, 3])),
            inliers_by_camera={name: sum(key[0] == name for key in best.inliers) for name in cameras})
        for other in candidates[1:]:
            if (len(other.inliers) == len(best.inliers) and
                other.reprojection_px-best.reprojection_px < quality["ambiguity_px"] and
                np.rad2deg(rotation_distance(best.pose[:3, :3], other.pose[:3, :3])) > quality["ambiguity_angle_deg"]):
                self.diagnostics["reason"] = "ambiguous_pose"
                for obs in observations:
                    obs.pose_reason = "ambiguous_pose"
                    obs.reprojection_px = best.errors[obs.key]
                return None, "ambiguous_pose"
        self.diagnostics["reason"] = "ok"
        for obs in observations:
            obs.reprojection_px = best.errors[obs.key]
            obs.pose_reason = "ok" if obs.key in best.inliers else "reprojection_outlier"
        return best, "ok"
