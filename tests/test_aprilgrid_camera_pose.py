"""Physical coordinate conventions and stale/measurement isolation of the pose row."""
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from ego2.aprilgrid.app import frame_record
from ego2.aprilgrid.camera_pose import camera_in_board, camera_pose_frame, configure_pose_viewer, pose_document
from ego2.aprilgrid.config import Board, load_config
from ego2.aprilgrid.pose import Measurement, Solver
from ego2.aprilgrid.temporal import Pipeline
from ego2.aprilgrid.tracking import Track
from ego2.aprilgrid.visualizer import Visualizer
from ego2.common.sources import Pair
from ego2.common.config import load_calibration, read_yaml


def from_rpy(angles):
    roll, pitch, yaw = np.deg2rad(angles)
    rx = cv2.Rodrigues(np.array([roll, 0., 0.]))[0]
    ry = cv2.Rodrigues(np.array([0., pitch, 0.]))[0]
    rz = cv2.Rodrigues(np.array([0., 0., yaw]))[0]
    return rz @ ry @ rx


class CameraPoseTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.cameras, self.board = load_config()
        self.solver = Solver(self.cfg, self.cameras, self.board)
        self.track = Track(self.cfg["tracking"])
        # Camera at (+20, +10, +50) cm in board coordinates, looking at the
        # board with its X axis along board Y. Translation cannot just negate.
        self.expected = np.eye(4)
        self.expected[:3, :3] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
        self.expected[:3, 3] = [.2, .1, .5]
        self.pose = np.linalg.inv(self.expected)

    def accept(self, stamp=1., mode="stereo", cameras=("cam0", "cam1")):
        inliers = frozenset((camera, id_) for camera in cameras for id_ in (0, 6, 35))
        self.track.update(stamp, Measurement(self.pose.copy(), .8, mode, inliers, {}), "ok")

    def snapshot(self, stamp):
        return camera_pose_frame(stamp, self.track, self.solver.transforms)

    def test_camera_origin_and_axes_in_board_frame(self):
        result = camera_in_board(self.pose)
        np.testing.assert_allclose(result["T_board_camera"], self.expected, atol=1e-12)
        np.testing.assert_allclose(result["position_m"], [.2, .1, .5])
        self.assertFalse(np.allclose(result["position_m"], -self.pose[:3, 3]))
        np.testing.assert_allclose(from_rpy(result["rpy_deg"]), self.expected[:3, :3], atol=1e-12)
        # A front-facing camera has a half turn around X, not zero attitude.
        front = np.diag([1., -1., -1., 1.])
        front[2, 3] = .5
        np.testing.assert_allclose(camera_in_board(front)["rpy_deg"], [180, 0, 0])

    def test_rotations_roundtrip_including_half_turn_and_gimbal_lock(self):
        angles = [[0, 0, 0], [180, 0, 0], [0, 180, 0], [0, 0, 180],
                  [23, -37, 152], [25, 90, -42], [-65, -90, 130]]
        for rpy in angles:
            with self.subTest(rpy=rpy):
                expected = from_rpy(rpy)
                t = np.eye(4)
                t[:3, :3] = expected.T
                result = camera_in_board(t)
                np.testing.assert_allclose(from_rpy(result["rpy_deg"]), expected, atol=1e-7)
                x, y, z, w = result["quaternion_xyzw"]
                quaternion_rotation = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
                np.testing.assert_allclose(quaternion_rotation, expected, atol=1e-7)
                self.assertEqual(result["rpy_singular"], abs(rpy[1]) == 90)

    def test_stereo_extrinsics_and_mono_support_both_optical_centres(self):
        self.accept(mode="mono", cameras=("cam0",))
        result = self.snapshot(1.)
        self.assertTrue(result["valid"])
        self.assertEqual(result["inliers_by_camera"], {"cam0": 3, "cam1": 0})
        p0, p1 = [np.asarray(result["cameras"][name]["T_board_camera"]) for name in ("cam0", "cam1")]
        np.testing.assert_allclose(p1, self.expected, atol=1e-12)
        # The same camera-to-camera rigid transform must hold in any board pose.
        np.testing.assert_allclose(np.linalg.inv(p0) @ p1, self.solver.transforms["cam0"], atol=1e-12)
        self.assertIn("via stereo extrinsics", pose_document(result, "cam1", self.cfg["viewer"]))

    def test_predictions_never_refresh_held_values_or_export_measurements(self):
        self.accept()
        self.track.velocity[:] = [1, 0, 0]
        for stamp, expected_status in ((1.1, "stale"), (1.2, "stale"), (1.301, "lost")):
            self.track.update(stamp, None, "insufficient_inliers", image_support=True)
            row = frame_record(Pair((), stamp, 0, 1), stamp, [], self.track, self.board, self.solver, 1)
            result = row["camera_pose"]
            self.assertFalse(result["valid"])
            self.assertEqual(result["status"], expected_status)
            self.assertEqual(result["measurement_time_s"], 1.)
            self.assertAlmostEqual(result["age_s"], stamp-1)
            self.assertIsNone(row["T_board_cam0"])
            self.assertIsNone(row["T_board_cam1"])
            self.assertTrue(all(v is None for v in result["cameras"].values()))
            if expected_status == "stale":
                np.testing.assert_allclose(result["held_cameras"]["cam1"]["T_board_camera"], self.expected, atol=1e-12)
                self.assertIn("LAST VALID ONLY", pose_document(result, "cam1", self.cfg["viewer"]))
            else:
                self.assertTrue(all(v is None for v in result["held_cameras"].values()))
                self.assertIn("`—`", pose_document(result, "cam1", self.cfg["viewer"]))
            json.dumps(row, allow_nan=False)
        self.accept(1.4)
        recovered = self.snapshot(1.4)
        self.assertTrue(recovered["valid"])
        self.assertEqual(recovered["age_s"], 0)

    def test_missing_and_rewound_data_do_not_reuse_future_pose(self):
        self.assertEqual(self.snapshot(0)["status"], "not_seen")
        self.accept(1)
        result = self.snapshot(.5)
        self.assertEqual(result["status"], "not_seen")
        self.assertIsNone(result["measurement_time_s"])
        self.assertTrue(all(v is None for v in result["held_cameras"].values()))

    def test_jump_rejection_is_not_an_accepted_camera_pose(self):
        self.accept()
        jumped = self.pose.copy()
        jumped[0, 3] += 1
        self.track.update(1.03, Measurement(jumped, .5, "stereo", frozenset(), {}), "ok")
        self.assertEqual(self.track.status, "pose_jump")
        result = self.snapshot(1.03)
        self.assertFalse(result["valid"])
        np.testing.assert_allclose(result["held_cameras"]["cam1"]["T_board_camera"], self.expected, atol=1e-12)

    def test_visualizer_overwrites_text_when_lost_at_same_image_timestamp(self):
        viewer = Visualizer.__new__(Visualizer)
        viewer.cfg, viewer.cameras, viewer.board = self.cfg, self.cameras, self.board
        viewer.transforms, viewer.recording = self.solver.transforms, MagicMock()
        images = (np.zeros((32, 32, 3), np.uint8),)*2
        self.accept()
        for stamp, state in ((1., "VALID - measured"), (1.1, "LAST VALID ONLY"), (1.4, "LOST")):
            if stamp > 1:
                self.track.update(stamp, None, "not_seen")
            viewer.recording.reset_mock()
            viewer.log(Pair(images, stamp, 0, 42), stamp, [], self.track, 1)
            viewer.recording.set_time.assert_any_call("capture_time", duration=stamp)
            viewer.recording.set_time.assert_any_call("frame", sequence=42)
            texts = [call.args[1].text.as_arrow_array().to_pylist()[0]
                     for call in viewer.recording.log.call_args_list if call.args[0].startswith("/camera_pose/")]
            self.assertEqual(len(texts), 2)
            self.assertTrue(all(state in text for text in texts))

    def test_old_viewer_profiles_get_defaults_and_invalid_options_rejected(self):
        self.assertEqual(configure_pose_viewer({})["pose_hold_s"], .3)
        for key, value in (("pose_hold_s", -.1), ("pose_hold_s", float("nan")),
                           ("pose_row_share", 1), ("pose_position_decimals", 2.5),
                           ("pose_angle_decimals", True)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                configure_pose_viewer({key: value})

    def test_real_recorded_sequence_camera_matrices_and_export_are_consistent(self):
        fixtures = Path(__file__).parent/"fixtures"/"aprilgrid"
        cameras = load_calibration(fixtures/"calibration.yaml")
        board = Board(read_yaml(fixtures/"target.yaml"))
        pipeline = Pipeline(self.cfg, cameras, board)
        cv2.setNumThreads(1)
        try:
            for index, number in enumerate((972, 974, 976, 978, 980)):
                images = tuple(cv2.imread(str(fixtures/f"{number}-{name}.jpg")) for name in ("cam0", "cam1"))
                stamp = index/15
                obs = pipeline.process(images, stamp)
                self.assertIsNotNone(pipeline.track.measurement)
                row = frame_record(Pair(images, stamp, 0, number), stamp, obs, pipeline.track,
                                   board, pipeline.solver, 1)
                for name in cameras:
                    actual = np.asarray(row[f"T_board_{name}"])
                    expected = np.linalg.inv(pipeline.solver.transforms[name] @ pipeline.track.measurement.pose)
                    np.testing.assert_allclose(actual, expected, atol=1e-10)
                    self.assertEqual(row["camera_pose"]["cameras"][name]["T_board_camera"], actual.tolist())
                json.dumps(row, allow_nan=False)
        finally:
            pipeline.close()


if __name__ == "__main__":
    unittest.main()
