import copy
import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from ego2.common.sources import Pair
from ego2.hand_pose.config import load_config as load_default, validate
from ego2.hand_pose.detector import Detector
from ego2.hand_pose.demo import DemoSource
from ego2.hand_pose.pose import Observation, Measurement, corners, solve, projections, rotation_distance, quaternion_xyzw
from ego2.hand_pose.tracking import Track


def load_config():
    """Keep legacy single-tag regressions while the shipped profile uses bundles."""
    cfg, cameras = load_default()
    for target in cfg["targets"]:
        target["id"] = target.pop("ids")[0]
        target["T_marker_endpoint"] = target.pop("T_wrist_endpoint")
    return cfg, cameras


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.cameras = load_config()
        self.detector = Detector(self.cfg, self.cameras)
        self.size = self.cfg["targets"][0]["size_m"]
        self.pose = np.eye(4)
        self.pose[:3, :3] = cv2.Rodrigues(np.array([0.2, -0.15, 0.1]))[0] @ np.diag([1., -1., -1.])
        self.pose[:3, 3] = [-0.025, 0.045, 0.28]

    def observations(self):
        result = []
        for cam, _, _, _, t in self.detector.views:
            o = Observation(cam.name, None, self.detector.K, t)
            o.pixels = projections(self.pose, corners(self.size), o)
            result.append(o)
        return result

    def test_stereo_pose_uses_correct_camera_direction_and_metric_scale(self):
        m, reason = solve(self.size, self.observations(), self.cfg["quality"])
        self.assertEqual(reason, "ok")
        self.assertEqual(m.mode, "stereo")
        np.testing.assert_allclose(m.T_head_marker, self.pose, atol=1e-6)
        self.assertLess(m.reprojection_px, 1e-5)

    def test_single_eye_requires_explicit_opt_in(self):
        observations = self.observations()[:1]
        m, reason = solve(self.size, observations, self.cfg["quality"])
        self.assertIsNone(m)
        self.assertEqual(reason, "need_both_cameras")
        m, reason = solve(self.size, observations, {**self.cfg["quality"], "require_stereo": False})
        self.assertEqual(reason, "ok")
        np.testing.assert_allclose(m.T_head_marker, self.pose, atol=1e-6)

    def test_stereo_mismatch_is_rejected_not_replaced_by_mono(self):
        observations = self.observations()
        observations[1].pixels += [25, 18]
        m, reason = solve(self.size, observations, {**self.cfg["quality"], "require_stereo": False})
        self.assertIsNone(m)
        self.assertEqual(reason, "inconsistent_geometry")

    def test_wrong_print_size_fails_stereo_geometry(self):
        m, _ = solve(self.size*2, self.observations(), self.cfg["quality"])
        self.assertIsNone(m)

    def test_small_or_nonfinite_corners_rejected(self):
        observations = self.observations()
        observations[0].pixels[:] = observations[0].pixels.mean(axis=0)
        self.assertEqual(solve(self.size, observations, self.cfg["quality"])[1], "marker_too_small")
        observations[0].pixels[0, 0] = np.nan
        self.assertEqual(solve(self.size, observations, self.cfg["quality"])[1], "invalid_corners")

    def test_quaternion_180_degree_convention(self):
        np.testing.assert_allclose(quaternion_xyzw(np.diag([1., -1., -1.])), [1, 0, 0, 0], atol=1e-6)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.cfg, _ = load_config()
        self.track = Track(self.cfg["targets"][0], self.cfg["tracking"])

    def measurement(self, x=0., angle=0.):
        p = np.eye(4)
        p[:3, 3] = [x, 0, 0.3]
        p[:3, :3] = cv2.Rodrigues(np.array([0., 0., angle]))[0]
        return Measurement(p, 0.2, "stereo")

    def test_loss_hides_pose_and_breaks_trace(self):
        self.track.update(0, self.measurement(), "ok")
        self.track.update(.03, self.measurement(.01), "ok")
        self.track.update(.06, None, "not_seen")
        self.assertIsNone(self.track.pose)
        self.assertEqual(len(self.track.path), 2)
        self.track.update(.09, self.measurement(.02), "ok")
        self.track.update(.12, self.measurement(.03), "ok")
        self.assertEqual([len(s) for s in self.track.strips()], [2, 2])

    def test_frame_gap_starts_new_segment_without_interpolation(self):
        self.track.update(0, self.measurement(), "ok")
        self.track.update(.8, self.measurement(.4), "ok")
        self.assertEqual(self.track.segment, 1)
        self.assertEqual(self.track.pose[0, 3], .4)
        self.assertEqual(self.track.strips(), [])

    def test_implausible_jump_is_invalid(self):
        self.track.update(0, self.measurement(), "ok")
        self.track.update(.03, self.measurement(.7), "ok")
        self.assertEqual(self.track.status, "motion_jump")
        self.assertIsNone(self.track.pose)
        self.assertEqual(len(self.track.path), 1)

    def test_filter_is_valid_rotation_and_endpoint_offset_composes_correctly(self):
        target = copy.deepcopy(self.cfg["targets"][0])
        target["T_marker_endpoint"][0][3] = .02
        track = Track(target, self.cfg["tracking"])
        track.update(0, self.measurement(angle=np.pi/2), "ok")
        np.testing.assert_allclose(track.pose[:3, 3], [0, .02, .3], atol=1e-8)
        track.update(.03, self.measurement(.01, np.pi/2+.1), "ok")
        r = track.pose[:3, :3]
        np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(r), 1.)
        self.assertLess(rotation_distance(r, track.raw_pose[:3, :3]), .1)

    def test_history_memory_is_bounded(self):
        settings = {**self.cfg["tracking"], "trajectory_points": 10, "pose_history_count": 3}
        track = Track(self.cfg["targets"][0], settings)
        for i in range(100):
            track.update(i*.1, self.measurement(), "ok")
        self.assertEqual(len(track.path), 10)
        self.assertEqual(len(track.history), 3)


class DetectorTests(unittest.TestCase):
    def test_actual_small_two_border_fisheye_tags_recover_both_poses(self):
        cfg, cams = load_config()
        source = DemoSource(cfg, cams)
        source.open()
        pair = source.read(threading.Event())
        detector = Detector(cfg, cams)
        results = detector.process(pair.images)
        for name, (measurement, reason) in results.items():
            self.assertEqual(reason, "ok", (name, detector.last_seen))
            truth = source.truth[name]
            self.assertLess(np.linalg.norm(measurement.T_head_marker[:3, 3]-truth[:3, 3]), .005)
            self.assertLess(np.rad2deg(rotation_distance(measurement.T_head_marker[:3, :3], truth[:3, :3])), 10)
        wrong_border = copy.deepcopy(cfg)
        wrong_border["border_bits"] = 1
        wrong = Detector(wrong_border, cams).process(pair.images)
        self.assertTrue(all(measurement is None for measurement, reason in wrong.values()))

    def test_blank_images_never_produce_measurements(self):
        cfg, cams = load_config()
        images = tuple(np.full((cams[c["camera"]].size[1], cams[c["camera"]].size[0], 3), 160, np.uint8) for c in cfg["channels"])
        result = Detector(cfg, cams).process(images)
        self.assertTrue(all(m is None for m, _ in result.values()))

    def test_marker_binding_validation(self):
        cfg, _ = load_config()
        cfg["targets"][1]["id"] = cfg["targets"][0]["id"]
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_viewer_clears_missing_current_pose_and_keeps_raw_views(self):
        from ego2.hand_pose.visualizer import Visualizer
        cfg, _ = load_config()
        viewer = Visualizer.__new__(Visualizer)
        viewer.cfg, viewer.axis = cfg, .03
        viewer.recording = MagicMock()
        tracks = {t["name"]: Track(t, cfg["tracking"]) for t in cfg["targets"]}
        images = (np.zeros((20, 20, 3), np.uint8),)*2
        pair = Pair(images, 0, 0, 1)
        viewer.log(pair, 0, tracks)
        paths = [call.args[0] for call in viewer.recording.log.call_args_list]
        self.assertIn("/head/hands/left/current", paths)
        self.assertIn("/rgb/cam0", paths)
        self.assertIn("/rgb/cam1", paths)

    def test_unconfirmed_ids_do_not_open_camera(self):
        from ego2.hand_pose.__main__ import main
        cfg, cams = load_config()
        cfg["markers_configured"] = False
        with patch("ego2.hand_pose.__main__.load_config", return_value=(cfg, cams)), patch("ego2.hand_pose.app.run") as run:
            self.assertEqual(main([]), 1)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
