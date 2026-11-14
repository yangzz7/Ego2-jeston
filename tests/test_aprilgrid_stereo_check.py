"""Detect circular validation and verify frame direction, sampling and reporting."""
import copy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ego2.aprilgrid.camera_pose import camera_pose_frame
from ego2.aprilgrid.config import load_config
from ego2.aprilgrid.detector import Observation
from ego2.aprilgrid.pose import Measurement, Solver, rotation_distance
from ego2.aprilgrid.stereo_check import StereoCheck, configure_stereo_check, print_report, relative_pose, representative_pose
from ego2.aprilgrid.tracking import Track
from ego2.common.geometry import project


class StereoCheckTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.cameras, self.board = load_config()
        self.cfg["stereo_check"]["min_accepted_samples"] = 2
        self.true_relative = self.cameras["cam1"].T_from_root @ np.linalg.inv(self.cameras["cam0"].T_from_root)
        self.pose = np.eye(4)
        self.pose[:3, :3] = cv2.Rodrigues(np.array([.2, -.3, .1]))[0] @ np.diag([1., -1., -1.])
        self.pose[:3, 3] = [.025, .015, .65]
        self.ids = (0, 5, 14, 21, 30, 35)

    def observations(self, ids=None):
        result = []
        for name in ("cam0", "cam1"):
            t = self.pose if name == "cam1" else np.linalg.inv(self.true_relative) @ self.pose
            camera = self.cameras[name]
            for id_ in self.ids if ids is None else ids:
                xyz = self.board.corners(id_) @ t[:3, :3].T+t[:3, 3]
                u, v, valid = project(camera, xyz)
                self.assertTrue(valid.all())
                pixels = np.c_[u, v]
                result.append(Observation(name, id_, pixels, pixels, pixels, camera.K, np.eye(4), 0))
        return result

    def snapshot(self, cameras=None, stamp=0., measured=True):
        cameras = cameras or self.cameras
        track = Track(self.cfg["tracking"])
        if measured:
            track.update(stamp, Measurement(self.pose, .1, "stereo", frozenset(), {}), "ok")
        return camera_pose_frame(stamp, track, Solver(self.cfg, cameras, self.board).transforms)

    def test_relative_direction_matches_kalibr_and_optical_centre(self):
        b_c1 = np.linalg.inv(self.pose)
        b_c0 = np.linalg.inv(np.linalg.inv(self.true_relative) @ self.pose)
        actual = relative_pose(b_c0, b_c1)
        np.testing.assert_allclose(actual, self.true_relative, atol=1e-12)
        origin_in_cam1 = actual @ [0, 0, 0, 1]
        np.testing.assert_allclose(origin_in_cam1[:3], self.true_relative[:3, 3])
        np.testing.assert_allclose(relative_pose(b_c1, b_c0), np.linalg.inv(actual), atol=1e-12)

    def test_wrong_supplied_extrinsics_pass_algebra_but_fail_independent_comparison(self):
        corrupted = self.true_relative.copy()
        corrupted[:3, :3] = cv2.Rodrigues(np.array([0., 0., np.deg2rad(3.)]))[0] @ corrupted[:3, :3]
        corrupted[:3, 3] += [.025, 0, 0]
        wrong_cameras = {**self.cameras, "cam1": replace(self.cameras["cam1"], T_from_root=corrupted)}
        check = StereoCheck(self.cfg, wrong_cameras, self.board)
        obs = self.observations()
        # Even observations rejected by the *joint* fit must remain eligible.
        for o in obs:
            o.pose_reason, o.reprojection_px = "reprojection_outlier", 100.
        check.observe(0., 1, obs, self.snapshot(wrong_cameras))
        report = check.finish()
        self.assertLess(report["algebra_check"]["max_errors"]["matrix_abs_error"], 1e-12)
        independent = report["independent_check"]
        self.assertEqual(independent["accepted_pairs"], 1)
        estimate = np.asarray(independent["estimate"]["matrix"])
        np.testing.assert_allclose(estimate, self.true_relative, atol=1e-6)
        self.assertAlmostEqual(independent["difference"]["translation_error_mm"], 25., places=3)
        self.assertAlmostEqual(independent["difference"]["rotation_error_deg"], 3., places=3)
        self.assertEqual(independent["status"], "limited_samples")
        # Offline solving cannot overwrite the live observations/diagnostics.
        self.assertTrue(all(o.pose_reason == "reprojection_outlier" and o.reprojection_px == 100 for o in obs))

    def test_each_eye_solves_without_stereo_extrinsics_or_joint_acceptance(self):
        check = StereoCheck(self.cfg, self.cameras, self.board)
        for name, solver in check.solvers.items():
            self.assertEqual(set(solver.cameras), {name})
            np.testing.assert_array_equal(solver.cameras[name].T_from_root, np.eye(4))
            np.testing.assert_array_equal(solver.transforms[name], np.eye(4))
        for i in range(2):
            check.observe(i*.6, i, self.observations(), self.snapshot(stamp=i*.6, measured=False))
        report = check.finish()
        self.assertEqual(report["algebra_check"]["frames"], 0)
        self.assertIsNone(report["algebra_check"]["max_errors"])
        self.assertEqual(report["independent_check"]["accepted_pairs"], 2)
        self.assertEqual(report["independent_check"]["status"], "available")
        self.assertLess(report["independent_check"]["difference"]["translation_error_mm"], .001)
        json.dumps(report, allow_nan=False)

    def test_never_pairs_different_frames_or_uses_flow_or_projected_corners(self):
        check = StereoCheck(self.cfg, self.cameras, self.board)
        empty = self.snapshot(measured=False)
        obs = self.observations()
        check.observe(0., 1, [o for o in obs if o.camera == "cam0"], empty)
        check.observe(.6, 2, [o for o in obs if o.camera == "cam1"], empty)
        check.observe(1.2, 3, [replace(o, source="board") for o in obs], empty)
        check.observe(1.8, 4, [replace(o, source="flow") for o in obs], empty)
        check.observe(2.4, 5, self.observations(ids=(0, 1, 2, 3, 4)), empty)
        self.assertEqual(check.samples, [])
        report = check.finish()
        self.assertEqual(report["independent_check"]["status"], "no_valid_samples")
        self.assertIsNone(report["independent_check"]["estimate"])
        lines = []
        print_report(report, lines.append)
        self.assertIn("不能据此判断参数正确", "\n".join(lines))

    def test_sampling_interval_and_bounded_reservoir_preserve_frame_pairs(self):
        self.cfg["stereo_check"].update(max_samples=3)
        check = StereoCheck(self.cfg, self.cameras, self.board)
        empty, obs = self.snapshot(measured=False), self.observations()
        for i in range(20):
            check.observe(i*.6, i, obs, empty)
            check.observe(i*.6+.01, 100+i, obs, empty)
        self.assertEqual(check.eligible_pairs, 20)
        self.assertEqual(len(check.samples), 3)
        self.assertTrue(any(s["frame"] > 3 for s in check.samples))
        self.assertTrue(all(s["frame"] < 20 and set(s["observations"]) == {"cam0", "cam1"} for s in check.samples))
        obs[0].pixels[:] += 100
        self.assertFalse(np.array_equal(check.samples[0]["observations"]["cam0"][0]["pixels"], obs[0].pixels))

    def test_rotation_summary_handles_wraparound_and_translation_outlier(self):
        poses = []
        for angle, x in ((179, .06), (-179, .061), (178, .3)):
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(np.array([0., 0., np.deg2rad(angle)]))[0]
            pose[0, 3] = x
            poses.append(pose)
        result = representative_pose(poses)
        self.assertAlmostEqual(result[0, 3], .061)
        self.assertLess(rotation_distance(result[:3, :3], poses[0][:3, :3]), 1e-7)

    def test_second_interrupt_keeps_completed_results_and_marks_partial(self):
        check = StereoCheck(self.cfg, self.cameras, self.board)
        for i in range(2):
            check.observe(i*.6, i, self.observations(), self.snapshot(measured=False))
        solve, calls = check._solve, 0
        def interrupted(name, tags):
            nonlocal calls
            calls += 1
            if calls > 2:
                raise KeyboardInterrupt
            return solve(name, tags)
        with patch.object(check, "_solve", side_effect=interrupted):
            report = check.finish()
        self.assertTrue(report["independent_check"]["interrupted"])
        self.assertEqual(report["independent_check"]["attempted_pairs"], 1)
        self.assertEqual(report["independent_check"]["accepted_pairs"], 1)
        json.dumps(report, allow_nan=False)

    def test_configuration_defaults_and_synthetic_reporting(self):
        self.assertTrue(configure_stereo_check({})["enabled"])
        for key, value in (("enabled", 1), ("sample_interval_s", 0), ("sample_interval_s", float("nan")),
                           ("max_samples", 0), ("min_tags_per_eye", 3)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                configure_stereo_check({"stereo_check": {key: value}})
        cfg = copy.deepcopy(self.cfg)
        cfg["stereo_check"]["enabled"] = False
        check = StereoCheck(cfg, self.cameras, self.board, synthetic_only=True)
        check.observe(0, 1, self.observations(), self.snapshot())
        report = check.finish()
        self.assertTrue(report["synthetic_only"])
        self.assertEqual(report["independent_check"]["status"], "disabled")
        lines = []
        print_report(report, lines.append)
        self.assertIn("不能用于验证真实相机参数", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
