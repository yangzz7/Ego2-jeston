import copy
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import yaml

from ego2.aprilgrid.config import Board, load_config
from ego2.aprilgrid.detector import Detector, Observation
from ego2.aprilgrid.demo import DemoSource
from ego2.aprilgrid.pose import Solver, Measurement, rotation_distance
from ego2.aprilgrid.tracking import Track
from ego2.common.geometry import project
from ego2.common.config import load_calibration
from ego2.common.sources import Pair
from ego2.aprilgrid.geometry import bearings, projection_jacobian

FIXTURES = Path(__file__).parent/"fixtures"/"aprilgrid"


class BoardTests(unittest.TestCase):
    def test_pdf_layout_and_rotated_decoder_corners(self):
        _, _, board = load_config()
        # Independent physical facts from the supplied PDF: ID 0 is bottom-left,
        # ID 35 top-right, and decoded corner zero is the printed bottom-right.
        np.testing.assert_allclose(board.centres[0], [-.17875, -.17875, 0])
        np.testing.assert_allclose(board.centres[35], [.17875, .17875, 0])
        np.testing.assert_allclose(board.corners(0)[0], [-.15125, -.20625, 0])
        self.assertEqual(board.border, 2)
        pose = board.tag_pose(np.eye(4), 0)
        np.testing.assert_allclose(pose[:3, :3], np.eye(3))
        np.testing.assert_allclose(pose[:3, 3], board.centres[0])

    def test_invalid_layout_and_dimensions_rejected(self):
        _, _, board = load_config()
        for edit in ({"tagSize": 0}, {"tagSpacing": float("nan")}, {"marker_rotation_deg": 45},
                     {"ids_by_row": [[0]*6]*6}, {"page_size_m": [.1, .1]}):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                Board({**board.data, **edit})


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.cameras, self.board = load_config()
        self.solver = Solver(self.cfg, self.cameras, self.board)
        self.pose = np.eye(4)
        self.pose[:3, :3] = cv2.Rodrigues(np.array([.2, -.22, .12]))[0] @ np.diag([1., -1., -1.])
        self.pose[:3, 3] = [.025, .015, .7]

    def observations(self, ids=(0, 5, 15, 30, 35), names=("cam0", "cam1")):
        observations = []
        for name in names:
            cam, t = self.cameras[name], self.solver.transforms[name]
            for marker_id in ids:
                pose = t @ self.pose
                xyz = self.board.corners(marker_id) @ pose[:3, :3].T + pose[:3, 3]
                u, v, _ = project(cam, xyz)
                pixels = np.c_[u, v]
                pinhole = xyz @ cam.K.T
                observations.append(Observation(name, marker_id, pixels, pixels.copy(),
                    pinhole[:, :2]/pinhole[:, 2:], cam.K, t, 0))
        return observations

    def test_joint_stereo_recovers_metric_board_and_camera_direction(self):
        m, reason = self.solver.solve(self.observations())
        self.assertEqual(reason, "ok")
        self.assertEqual(m.mode, "stereo")
        np.testing.assert_allclose(m.pose, self.pose, atol=1e-6)
        self.assertLess(m.reprojection_px, 1e-5)
        for name in self.cameras:
            tag = self.solver.transforms[name] @ self.board.tag_pose(m.pose, 0)
            expected = self.solver.transforms[name] @ self.pose @ np.array([
                [1, 0, 0, -.17875], [0, 1, 0, -.17875], [0, 0, 1, 0], [0, 0, 0, 1]])
            np.testing.assert_allclose(tag, expected, atol=1e-6)

    def test_mono_and_partial_board(self):
        m, reason = self.solver.solve(self.observations(ids=(0, 1, 6), names=("cam0",)))
        self.assertEqual(reason, "ok")
        self.assertEqual(m.mode, "mono")
        np.testing.assert_allclose(m.pose, self.pose, atol=1e-6)

    def test_one_marker_can_initialize(self):
        m, reason = self.solver.solve(self.observations(ids=(15,)))
        self.assertEqual(reason, "ok")
        np.testing.assert_allclose(m.pose, self.pose, atol=1e-5)

    def test_wrong_stereo_correspondence_cannot_silently_fall_back_to_mono(self):
        obs = self.observations()
        for o in obs:
            if o.camera == "cam1":
                o.pixels += [35, 25]
                o.view_pixels += [35, 25]
        m, _ = self.solver.solve(obs)
        self.assertIsNone(m)

    def test_an_outlier_does_not_move_all_tag_axes(self):
        obs = self.observations()
        obs[0].pixels += [35, -20]
        obs[0].view_pixels += [35, -20]
        m, reason = self.solver.solve(obs)
        self.assertEqual(reason, "ok")
        self.assertNotIn(obs[0].key, m.inliers)
        np.testing.assert_allclose(m.pose, self.pose, atol=1e-6)

    def test_refit_reclassifies_tags_and_retains_best_verified_consensus(self):
        obs = self.observations()
        masks = [{0,1,2,3,5,6,7,8}, {0,1,2,5,6,7,9}, {0,1,2,3,5,6,7,9}]
        calls = []
        def refine(initial, subset, iterations=10):
            calls.append(len(subset))
            pose = self.pose.copy()
            pose[0, 3] = min(len(calls), 3)*.01
            return pose
        def residual(pose, packed, count):
            stage = int(round(pose[0, 3]/.01))-1
            e = np.zeros((len(obs), 4, 2))
            e[:, :, 0] = np.array([1-.2*stage if i in masks[stage] else 4 for i in range(len(obs))])[:, None]
            return e.ravel()
        with patch.object(self.solver, "initial_poses", return_value=[self.pose]), \
             patch.object(self.solver, "refine", side_effect=refine), \
             patch.object(self.solver, "residual", side_effect=residual):
            m, reason = self.solver.solve(obs)
        self.assertEqual(reason, "ok")
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(m.inliers, frozenset(obs[i].key for i in masks[2]))
        self.assertAlmostEqual(m.reprojection_px, .6)

    def test_analytic_jacobian_matches_finite_difference(self):
        from ego2.aprilgrid.pose import transform
        observations = self.observations()
        vector = np.r_[cv2.Rodrigues(self.pose[:3, :3])[0].ravel(), self.pose[:3, 3]]
        packed = self.solver.pack(observations)
        jac = self.solver.jacobian(vector, packed, len(observations))
        numeric = []
        for axis in range(6):
            step = np.eye(6)[axis]*1e-6
            numeric.append((self.solver.residual(transform(vector+step), packed, len(observations))-
                            self.solver.residual(transform(vector-step), packed, len(observations)))/2e-6)
        np.testing.assert_allclose(jac, np.array(numeric).T, atol=1e-5, rtol=1e-6)
        cam = self.cameras["cam1"]
        # Optical axis and >90-degree rays must follow the same extended model.
        xyz = np.array([[0., 0., .5], [.7, .2, -.1], [-.7, -.2, -.1]])
        analytic = projection_jacobian(cam, xyz)
        numeric = []
        for axis in range(3):
            step = np.eye(3)[axis]*1e-6
            a = np.array(project(cam, xyz+step)[:2]).T
            b = np.array(project(cam, xyz-step)[:2]).T
            numeric.append((a-b)/2e-6)
        np.testing.assert_allclose(analytic, np.stack(numeric, axis=-1), atol=1e-5, rtol=1e-6)
        uv = np.array(project(cam, xyz)[:2]).T
        np.testing.assert_allclose(bearings(cam, uv), xyz/np.linalg.norm(xyz, axis=1)[:, None], atol=1e-8)

    def test_wrong_size_rejected_by_stereo(self):
        self.solver.board = Board({**self.board.data, "tagSize": .08, "page_size_m": [.8, .8]})
        m, _ = self.solver.solve(self.observations())
        self.assertIsNone(m)

    def test_track_loss_jump_and_reacquisition(self):
        track = Track(self.cfg["tracking"])
        m = Measurement(self.pose, 0, "stereo", frozenset(), {})
        track.update(0, m, "ok")
        track.update(.03, None, "not_seen")
        self.assertIsNone(track.measurement)
        self.assertIsNotNone(track.prior(.04))
        jump = self.pose.copy()
        jump[0, 3] += 1
        track.update(.04, Measurement(jump, 0, "stereo", frozenset(), {}), "ok")
        self.assertEqual(track.status, "pose_jump")
        self.assertIsNone(track.measurement)
        self.assertIsNone(track.prior(.4))
        track.update(.4, Measurement(jump, 0, "mono", frozenset(), {}), "ok")
        self.assertEqual(track.status, "mono")


class ImagePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.cfg, cls.cameras, cls.board = load_config()
        # Keep the original separated-tag pattern as an independent regression;
        # the exact PDF, including diagonal squares, is exercised below.
        plain = Board({**cls.board.data, "corner_squares": False})
        cls.source = DemoSource(cls.cfg, cls.cameras, plain)
        cls.source.open()
        cls.detector = Detector(cls.cfg, cls.cameras, cls.board)
        cls.solver = Solver(cls.cfg, cls.cameras, cls.board)

    @classmethod
    def tearDownClass(cls):
        cls.detector.close()
        cls.source.close()

    def test_render_decode_pose_and_blank_images(self):
        pair = self.source.read(threading.Event())
        obs = self.detector.observe(pair.images)
        for camera in self.cameras:
            ids = [o.marker_id for o in obs if o.camera == camera]
            self.assertGreaterEqual(len(ids), 34)
            self.assertEqual(len(ids), len(set(ids)))
        m, reason = self.solver.solve(obs)
        self.assertEqual(reason, "ok")
        self.assertLess(np.linalg.norm(m.pose[:3, 3]-self.source.truth[:3, 3]), .003)
        self.assertLess(np.rad2deg(rotation_distance(m.pose[:3, :3], self.source.truth[:3, :3])), 1)
        blanks = tuple(np.full_like(image, 160) for image in pair.images)
        self.assertEqual(self.detector.observe(blanks), [])

    def test_peripheral_views_find_board_outside_central_view(self):
        pose = self.source.pose_at(0)
        # Place the board at ~60 degrees to the right and face it toward the camera.
        turn = cv2.Rodrigues(np.array([0., np.deg2rad(60), 0.]))[0]
        pose[:3, :3] = turn @ pose[:3, :3]
        pose[:3, 3] = turn @ np.array([0., 0., .65])
        images = tuple(self.source.render(self.cameras[ch["camera"]], pose) for ch in self.cfg["channels"])
        obs = self.detector.observe(images)
        self.assertGreater(len(obs), 8)
        self.assertTrue(any(o.view_index != 0 for o in obs))
        m, reason = self.solver.solve(obs)
        self.assertEqual(reason, "ok")
        self.assertLess(np.linalg.norm(m.pose[:3, 3]-pose[:3, 3]), .005)

    def test_visualizer_clears_overlays_even_without_detections(self):
        from ego2.aprilgrid.visualizer import Visualizer, build_blueprint
        viewer = Visualizer.__new__(Visualizer)
        viewer.cfg, viewer.cameras, viewer.board = self.cfg, self.cameras, self.board
        viewer.transforms = self.solver.transforms
        viewer.recording = MagicMock()
        pair = self.source.read(threading.Event())
        obs = self.detector.observe(pair.images)
        track = Track(self.cfg["tracking"])
        track.update(0, *self.solver.solve(obs))
        viewer.log(pair, 0, obs, track, 20)
        viewer.recording.reset_mock()
        track.update(.03, None, "not_seen")
        viewer.log(pair, .03, [], track, 10)
        paths = [call.args[0] for call in viewer.recording.log.call_args_list]
        for camera in self.cameras:
            self.assertIn(f"/tracking/{camera}/overlays", paths)
            self.assertIn(f"/raw/{camera}/image", paths)
            self.assertIn(f"/tracking/{camera}/image", paths)
        self.assertFalse(any("tag_" in path for path in paths))
        self.assertIsNotNone(build_blueprint(self.cfg))


class RecordedAndPdfRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.cfg, _, _ = load_config()
        cls.cameras = load_calibration(FIXTURES/"calibration.yaml")
        cls.board = Board(yaml.safe_load((FIXTURES/"target.yaml").read_text()))
        cls.detector = Detector(cls.cfg, cls.cameras, cls.board)
        cls.solver = Solver(cls.cfg, cls.cameras, cls.board)

    @classmethod
    def tearDownClass(cls):
        cls.detector.close()

    def assert_verified(self, obs, measurement):
        self.assertIsNotNone(measurement, self.solver.diagnostics)
        packed = self.solver.pack(obs)
        error = self.solver.residual(measurement.pose, packed, len(obs)).reshape(-1, 4, 2)
        selected = [i for i, o in enumerate(obs) if o.key in measurement.inliers]
        norms = np.linalg.norm(error[selected], axis=2)
        self.assertTrue(np.all(np.sqrt(np.mean(norms**2, axis=1)) <= 2.5))
        self.assertTrue(np.all(norms <= 5))
        self.assertGreaterEqual(len(selected), np.ceil(.6*len(obs)))
        self.assertEqual({obs[i].camera for i in selected}, {"cam0", "cam1"})

    def test_real_far_oblique_and_close_recorded_frames(self):
        for frame in [548, 617, 972, 1555, 2322, 2468]:
            with self.subTest(frame=frame):
                images = [cv2.imread(str(FIXTURES/f"{frame}-{name}.jpg")) for name in ["cam0", "cam1"]]
                obs = self.detector.observe(images)
                m, reason = self.solver.solve(obs)
                self.assertEqual(reason, "ok", self.solver.diagnostics)
                self.assert_verified(obs, m)
                self.assertTrue(all(o.reprojection_px is not None for o in obs))

    def test_original_rejected_corners_survive_iterative_filtering(self):
        row = json.loads((FIXTURES/"617-original-corners.json").read_text())
        obs = []
        for tag in row["tags"]:
            cam = self.cameras[tag["camera"]]
            pixels = np.asarray(tag["pixels"])
            rays = bearings(cam, pixels)
            uv = rays @ cam.K.T
            obs.append(Observation(cam.name, tag["id"], pixels, pixels,
                                   uv[:, :2]/uv[:, 2:], cam.K, self.solver.transforms[cam.name], 0))
        m, reason = self.solver.solve(obs)
        self.assertEqual(reason, "ok", self.solver.diagnostics)
        self.assert_verified(obs, m)

    def test_actual_pdf_texture_with_close_blur_and_oblique_view(self):
        source = DemoSource(self.cfg, self.cameras, self.board)
        source.texture = cv2.imread(str(FIXTURES/"calib_io_500x500_6x6_55.png"), cv2.IMREAD_GRAYSCALE)
        source.open()
        try:
            for depth, angle, blur in [(.22, 0, 1.), (.5, 30, .6), (.65, 60, 0)]:
                with self.subTest(depth=depth, angle=angle, blur=blur):
                    pose = source.pose_at(0)
                    turn = cv2.Rodrigues(np.array([0., np.deg2rad(angle), 0.]))[0]
                    pose[:3, :3] = turn @ pose[:3, :3]
                    pose[:3, 3] = turn @ np.array([0., 0., depth])
                    images = [source.render(self.cameras[name], pose) for name in ["cam0", "cam1"]]
                    if blur:
                        images = [cv2.GaussianBlur(image, (0, 0), blur) for image in images]
                    obs = self.detector.observe(images)
                    m, reason = self.solver.solve(obs)
                    self.assertEqual(reason, "ok", self.solver.diagnostics)
                    self.assert_verified(obs, m)
                    self.assertLess(np.linalg.norm(m.pose[:3, 3]-pose[:3, 3]), .003)
                    self.assertLess(np.rad2deg(rotation_distance(m.pose[:3, :3], pose[:3, :3])), 1.)
        finally:
            source.close()

    def test_rejected_pose_records_errors_but_no_pose_or_axes(self):
        from ego2.aprilgrid.app import frame_record
        from ego2.aprilgrid.visualizer import Visualizer
        images = [cv2.imread(str(FIXTURES/f"972-{name}.jpg")) for name in ["cam0", "cam1"]]
        obs = self.detector.observe(images)
        for o in obs:
            if o.camera == "cam1":
                o.pixels += [35, 25]
        m, reason = self.solver.solve(obs)
        self.assertIsNone(m)
        self.assertIn(reason, {"insufficient_inliers", "stereo_inconsistent"})
        track = Track(self.cfg["tracking"])
        track.update(0, m, reason)
        pair = Pair(images, 0, 0, 972)
        row = frame_record(pair, 0, obs, track, self.board, self.solver, 1)
        self.assertIsNone(row["T_reference_board"])
        self.assertTrue(all("T_camera_tag" not in t for t in row["tags"]))
        self.assertTrue(any(t.get("reprojection_px", 0) > 2.5 for t in row["tags"]))
        viewer = Visualizer.__new__(Visualizer)
        viewer.cfg, viewer.cameras, viewer.board = self.cfg, self.cameras, self.board
        viewer.transforms, viewer.recording = self.solver.transforms, MagicMock()
        viewer.log(pair, 0, obs, track, 1, self.solver.diagnostics)
        paths = [call.args[0] for call in viewer.recording.log.call_args_list]
        import rerun as rr
        axes = [call for call in viewer.recording.log.call_args_list if call.args[0].endswith("/axes")]
        self.assertTrue(axes)
        self.assertTrue(all(isinstance(call.args[1], rr.Clear) for call in axes))
        self.assertIn("/tracking/cam0/overlays/ids", paths)
        self.assertLess(len(paths), 20)


class CliTests(unittest.TestCase):
    def test_script_from_other_directory_records_complete_demo(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)/"recording"
            result = subprocess.run([str(root/"run_aprilgrid.sh"), "--demo", "--frames", "2",
                "--headless", "--size-mm", "50", "--session", str(session)], cwd=temporary,
                capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            meta = json.loads((session/"session.json").read_text())
            self.assertTrue(meta["finished"])
            self.assertTrue(meta["synthetic_only"])
            self.assertEqual(meta["config"]["target_parameters"]["tagSize"], .05)
            self.assertEqual(yaml.safe_load((session/"target.yaml").read_text())["tagSize"], .05)
            rows = [json.loads(row) for row in (session/"poses.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["status"] == "stereo" for row in rows))
            self.assertGreater((session/"viewer.rrd").stat().st_size, 1000)
            stereo = json.loads((session/"stereo_check.json").read_text())
            self.assertTrue(stereo["synthetic_only"])
            self.assertEqual(stereo["algebra_check"]["frames"], 2)
            self.assertIn("双目相对位姿与参数对照", result.stdout)
            self.assertIn("不能用于验证真实相机参数", result.stdout)


if __name__ == "__main__":
    unittest.main()
