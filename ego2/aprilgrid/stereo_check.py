"""End-of-session stereo extrinsic comparison from independent monocular fits.

The live joint poses already contain the supplied stereo extrinsics. Their
relative transform is an algebra check only. Independent fitting uses fresh
decoded native corners, each eye's intrinsics, and physical board geometry;
it never seeds, filters or optimizes against the supplied stereo transform.
"""
from collections import Counter
from dataclasses import replace
import math
import time

import cv2
import numpy as np

from .camera_pose import camera_in_board
from .detector import Observation
from .pose import Solver, rotation_distance


DEFAULTS = {"enabled": True, "sample_interval_s": .5, "max_samples": 200,
            "min_tags_per_eye": 4, "min_accepted_samples": 10}


def configure_stereo_check(cfg):
    settings = cfg.setdefault("stereo_check", {})
    for key, value in DEFAULTS.items():
        settings.setdefault(key, value)
    if type(settings["enabled"]) is not bool:
        raise ValueError("stereo_check.enabled must be boolean")
    interval = settings["sample_interval_s"]
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("stereo_check.sample_interval_s must be finite and positive")
    for key, low, high in (("max_samples", 1, 2000), ("min_tags_per_eye", 4, 1000),
                           ("min_accepted_samples", 1, 2000)):
        value = settings[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"stereo_check.{key} must be an integer {low}..{high}")
    return settings


def relative_pose(T_board_cam0, T_board_cam1):
    """T_cam1_cam0 maps cam0 points into cam1 coordinates."""
    return np.linalg.inv(T_board_cam1) @ T_board_cam0


def difference(estimate, reference):
    delta = (estimate[:3, 3]-reference[:3, 3])*1000
    return {"translation_delta_mm_in_cam1": delta.tolist(),
            "translation_error_mm": float(np.linalg.norm(delta)),
            "rotation_error_deg": float(np.rad2deg(rotation_distance(reference[:3, :3], estimate[:3, :3]))),
            "baseline_mm": float(np.linalg.norm(estimate[:3, 3])*1000),
            "baseline_delta_mm": float((np.linalg.norm(estimate[:3, 3])-np.linalg.norm(reference[:3, 3]))*1000)}


def describe(pose):
    # camera_in_board describes the inverse of its input; here describe this
    # relative transform itself with the same RPY and quaternion convention.
    values = camera_in_board(np.linalg.inv(pose))
    return {"matrix": pose.tolist(), "translation_m": values["position_m"],
            "rpy_deg": values["rpy_deg"], "quaternion_xyzw": values["quaternion_xyzw"],
            "baseline_mm": float(np.linalg.norm(pose[:3, 3])*1000)}


def distribution(values):
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def representative_pose(poses):
    """Translation component median and rotation geodesic medoid, no reference."""
    stack = np.asarray(poses)
    rotations = stack[:, :3, :3]
    traces = np.einsum("aij,bij->ab", rotations, rotations)
    angles = np.arccos(np.clip((traces-1)/2, -1, 1))
    result = np.eye(4)
    result[:3, :3] = rotations[np.argmin(angles.sum(axis=1))]
    result[:3, 3] = np.median(stack[:, :3, 3], axis=0)
    return result


class StereoCheck:
    def __init__(self, cfg, cameras, board, *, synthetic_only=False, calibration_file=None):
        self.settings = configure_stereo_check(cfg)
        self.board = board
        self.synthetic_only = synthetic_only
        self.reference = cameras["cam1"].T_from_root @ np.linalg.inv(cameras["cam0"].T_from_root)
        self.calibration_path = calibration_file or cfg["calibration"]["file"]
        self.solvers = {}
        for name in ("cam0", "cam1"):
            # Remove all stereo extrinsics, even from the camera objects. These
            # solvers estimate T_cam_i_board directly, without a joint prior.
            mono_camera = replace(cameras[name], T_from_root=np.eye(4))
            self.solvers[name] = Solver({**cfg, "reference": name}, {name: mono_camera}, board)
        self.samples = []
        self.rng = np.random.default_rng(0)
        self.last_sample_time = None
        self.eligible_pairs = self.frames = self.algebra_frames = 0
        self.skipped = Counter()
        self.last_relative = None
        self.algebra_max = {"translation_error_mm": 0., "rotation_error_deg": 0., "matrix_abs_error": 0.}

    def _spans_board(self, ids):
        if len(ids) < self.settings["min_tags_per_eye"]:
            return False
        points = np.array([self.board.centres[i][:2] for i in ids], np.float32)
        # Reject a row/column/diagonal of tag centres; require a 2D footprint.
        return cv2.contourArea(cv2.convexHull(points)) >= .49*self.board.pitch**2

    def observe(self, stamp, frame_id, decoded, camera_pose):
        """Cheap live collection; independent PnP runs only during finish()."""
        if not self.settings["enabled"]:
            return
        self.frames += 1
        if camera_pose["valid"]:
            poses = camera_pose["cameras"]
            relative = relative_pose(np.asarray(poses["cam0"]["T_board_camera"]),
                                     np.asarray(poses["cam1"]["T_board_camera"]))
            delta = difference(relative, self.reference)
            self.algebra_frames += 1
            self.last_relative = relative
            for key in ("translation_error_mm", "rotation_error_deg"):
                self.algebra_max[key] = max(self.algebra_max[key], delta[key])
            self.algebra_max["matrix_abs_error"] = max(self.algebra_max["matrix_abs_error"],
                float(np.max(np.abs(relative-self.reference))))
        if not decoded:
            return
        if (self.last_sample_time is not None and
            0 <= stamp-self.last_sample_time < self.settings["sample_interval_s"]):
            self.skipped["sample_interval"] += 1
            return
        self.last_sample_time = stamp
        eyes = {}
        for name in ("cam0", "cam1"):
            usable = {o.marker_id: o for o in decoded if o.camera == name and
                      o.source == "decoded" and o.reason == "ok"}
            if not self._spans_board(list(usable)):
                self.skipped[name+":insufficient_tags_or_2d_coverage"] += 1
                return
            eyes[name] = [{"id": id_, "pixels": o.pixels.copy().tolist()} for id_, o in sorted(usable.items())]
        sample = {"frame": int(frame_id), "time_s": float(stamp), "observations": eyes}
        self.eligible_pairs += 1
        if len(self.samples) < self.settings["max_samples"]:
            self.samples.append(sample)
        else:
            # Bounded memory, uniform reservoir across the whole session.
            index = int(self.rng.integers(self.eligible_pairs))
            if index < len(self.samples):
                self.samples[index] = sample

    def _solve(self, name, tags):
        solver = self.solvers[name]
        camera = solver.cameras[name]
        observations = []
        for tag in tags:
            pixels = np.asarray(tag["pixels"], float)
            observations.append(Observation(name, tag["id"], pixels, pixels, pixels,
                                            camera.K, np.eye(4), 0))
        measurement, reason = solver.solve(observations)  # no prior
        if measurement is not None and not self._spans_board([key[1] for key in measurement.inliers]):
            return None, "insufficient_inlier_2d_coverage"
        return measurement, reason

    def finish(self, progress=None):
        started = time.perf_counter()
        results, estimates, accepted, rejected = [], [], [], Counter()
        interrupted = False
        if self.settings["enabled"]:
            try:
                for index, sample in enumerate(sorted(self.samples, key=lambda s: s["time_s"])):
                    row = {**sample, "eyes": {}}
                    measurements = {}
                    for name in ("cam0", "cam1"):
                        try:
                            m, reason = self._solve(name, sample["observations"][name])
                        except (cv2.error, np.linalg.LinAlgError):
                            m, reason = None, "numerical_failure"
                        row["eyes"][name] = {"status": reason, "valid": m is not None,
                            "T_camera_board": m.pose.tolist() if m else None,
                            "reprojection_px": m.reprojection_px if m else None,
                            "inlier_ids": sorted(key[1] for key in m.inliers) if m else []}
                        measurements[name] = m
                        if m is None:
                            rejected[name+":"+reason] += 1
                    row["valid"] = all(m is not None for m in measurements.values())
                    if row["valid"]:
                        a, b = measurements["cam0"].pose, measurements["cam1"].pose
                        relative = b @ np.linalg.inv(a)
                        row["T_cam1_cam0"] = relative.tolist()
                        row["difference"] = difference(relative, self.reference)
                        estimates.append(relative)
                        accepted.append(row)
                    results.append(row)
                    if progress is not None and (index+1) % 25 == 0:
                        progress(f"[stereo-check] 已独立求解 {index+1}/{len(self.samples)} 组，双眼有效 {len(accepted)} 组", flush=True)
            except KeyboardInterrupt:
                interrupted = True
        independent = {"status": "disabled" if not self.settings["enabled"] else
                       "no_valid_samples" if not accepted else
                       "limited_samples" if len(accepted) < self.settings["min_accepted_samples"] else "available",
                       "interrupted": interrupted, "attempted_pairs": len(results), "accepted_pairs": len(accepted),
                       "rejection_counts": dict(rejected), "estimate": None,
                       "method": "separate monocular fits, no stereo prior; translation component median + rotation geodesic medoid",
                       "uses": "fresh decoded corners, per-eye intrinsics/distortion and board dimensions; no agreement-based rejection"}
        if estimates:
            representative = representative_pose(estimates)
            normals = np.array([np.asarray(r["eyes"]["cam1"]["T_camera_board"])[:3, 2] for r in accepted])
            distances = [np.linalg.norm(np.asarray(r["eyes"]["cam1"]["T_camera_board"])[:3, 3]) for r in accepted]
            independent.update(estimate=describe(representative), inverse_estimate=describe(np.linalg.inv(representative)),
                difference=difference(representative, self.reference),
                per_sample_translation_error_mm=distribution([r["difference"]["translation_error_mm"] for r in accepted]),
                per_sample_rotation_error_deg=distribution([r["difference"]["rotation_error_deg"] for r in accepted]),
                translation_scatter_mm=distribution([np.linalg.norm(t[:3, 3]-representative[:3, 3])*1000 for t in estimates]),
                rotation_scatter_deg=distribution([np.rad2deg(rotation_distance(t[:3, :3], representative[:3, :3])) for t in estimates]),
                board_normal_span_deg=float(np.rad2deg(np.arccos(np.clip(np.min(normals @ normals.T), -1, 1)))),
                board_distance_range_m=[float(min(distances)), float(max(distances))])
        return {"version": 1, "synthetic_only": self.synthetic_only, "calibration_file": str(self.calibration_path),
                "direction": "T_cam1_cam0 maps cam0 points into cam1; translation is cam0 optical centre in cam1",
                "reference": describe(self.reference), "inverse_reference": describe(np.linalg.inv(self.reference)),
                "algebra_check": {"frames": self.algebra_frames, "max_errors": self.algebra_max if self.algebra_frames else None,
                    "last_relative": describe(self.last_relative) if self.last_relative is not None else None,
                    "independent_validation": False},
                "collection": {"frames": self.frames, "eligible_pairs": self.eligible_pairs,
                    "stored_pairs": len(self.samples), "skipped": dict(self.skipped), "settings": dict(self.settings)},
                "independent_check": independent, "samples": results,
                "postprocess_seconds": time.perf_counter()-started}


def report_overview(report):
    independent = report["independent_check"]
    return {"status": independent["status"], "accepted_pairs": independent["accepted_pairs"],
            "interrupted": independent["interrupted"], "synthetic_only": report["synthetic_only"],
            "difference": independent.get("difference")}


def print_report(report, output=print):
    def show(title, pose):
        output(title)
        output(np.array2string(np.asarray(pose["matrix"]), precision=8, suppress_small=True))
        xyz = np.asarray(pose["translation_m"])*1000
        output(f"  XYZ (mm): {xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f} | "
               f"RPY (deg): {', '.join(f'{a:+.4f}' for a in pose['rpy_deg'])} | 基线: {pose['baseline_mm']:.3f} mm")
    output("\n[双目相对位姿与参数对照]")
    output("方向 T_cam1_cam0：将 C1/cam0 的坐标转换到 C2/cam1；XYZ 是 cam0 光心在 cam1 中的位置。")
    output(f"参数文件：{report['calibration_file']}")
    show("参数参考 T_cam1_cam0：", report["reference"])
    algebra = report["algebra_check"]
    if algebra["frames"]:
        show("由界面两眼板坐标位姿计算：inverse(T_board_cam1) @ T_board_cam0", algebra["last_relative"])
        errors = algebra["max_errors"]
        output(f"  {algebra['frames']} 个有效帧的最大数值差：平移 {errors['translation_error_mm']:.3e} mm，"
               f"旋转 {errors['rotation_error_deg']:.3e} deg，矩阵元素 {errors['matrix_abs_error']:.3e}")
    else:
        output("本次没有有效的界面位姿，无法进行坐标变换一致性检查。")
    output("上述界面位姿已使用参数中的双目外参，因此这里只验证计算一致性，不代表独立验证外参。")
    check = report["independent_check"]
    output(f"\n独立单眼求解：保留 {report['collection']['stored_pairs']} 组，已求解 {check['attempted_pairs']} 组，"
           f"双眼均有效 {check['accepted_pairs']} 组。")
    if report["synthetic_only"]:
        output("本次为合成图像，仅验证计算流程，不能用于验证真实相机参数。")
    if check["interrupted"]:
        output("退出后求解被中断，以下只包含已完成的样本。")
    if check["estimate"] is not None:
        show("独立估计 T_cam1_cam0 = T_cam1_board @ inverse(T_cam0_board)：", check["estimate"])
        show("反方向 T_cam0_cam1（cam1 光心在 cam0 中）：", check["inverse_estimate"])
        delta = check["difference"]
        output(f"相对参数的差值：平移 {delta['translation_error_mm']:.3f} mm，旋转 {delta['rotation_error_deg']:.4f} deg，"
               f"基线差 {delta['baseline_delta_mm']:+.3f} mm")
        output("  平移分量差 ΔXYZ (mm, cam1)："+", ".join(f"{x:+.3f}" for x in delta["translation_delta_mm_in_cam1"]))
        for key, title, unit in (("per_sample_translation_error_mm", "各样本相对参数的平移差", "mm"),
                                 ("per_sample_rotation_error_deg", "各样本相对参数的旋转差", "deg"),
                                 ("translation_scatter_mm", "样本相对汇总值的平移离散", "mm"),
                                 ("rotation_scatter_deg", "样本相对汇总值的旋转离散", "deg")):
            stats = check[key]
            output(f"  {title}：中位数 {stats['median']:.3f}，P95 {stats['p95']:.3f}，最大 {stats['max']:.3f} {unit}")
        near, far = check["board_distance_range_m"]
        output(f"  板法向覆盖角 {check['board_normal_span_deg']:.1f} deg，板中心距离范围 {near:.3f}–{far:.3f} m")
        if check["status"] == "limited_samples":
            output("有效样本数量不足，当前估计仅供参考。")
    elif check["status"] == "disabled":
        output("独立对照已在配置中关闭。")
    else:
        output("有效样本不足，无法独立比较外参；不能据此判断参数正确。")
    if check["rejection_counts"]:
        output("独立求解拒绝原因："+str(check["rejection_counts"]))
    output("独立结果仍依赖各眼内参、畸变和实际标定板尺寸；报告给出差值与离散度，不自动修改参数或判定精度达标。")
