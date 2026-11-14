"""Temporal regressions use the actual PDF texture and known moving geometry."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from ego2.aprilgrid.app import frame_record
from ego2.aprilgrid.config import Board, load_config
from ego2.aprilgrid.demo import DemoSource
from ego2.aprilgrid.pose import Measurement, rotation_distance
from ego2.aprilgrid.temporal import Pipeline
from ego2.aprilgrid.tracking import Track, display_source
from ego2.aprilgrid.visualizer import Visualizer
from ego2.common.sources import Pair
from ego2.common.config import load_calibration, read_yaml

FIXTURES = Path(__file__).parent/"fixtures"/"aprilgrid"


class TemporalImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.cfg, cls.cameras, cls.board = load_config()
        source = DemoSource(cls.cfg, cls.cameras, cls.board)
        source.texture = cv2.imread(str(FIXTURES/"calib_io_500x500_6x6_55.png"), cv2.IMREAD_GRAYSCALE)
        source.open()
        cls.images, cls.truth = [], []
        for i in range(5):
            pose = source.pose_at(0)
            pose[:3, 3] += [i*.002, 0, 0]
            cls.truth.append(pose)
            cls.images.append(tuple(source.render(cls.cameras[name], pose) for name in ("cam0", "cam1")))
        source.close()

    def setUp(self):
        self.pipeline = Pipeline(copy.deepcopy(self.cfg), self.cameras, self.board)
        self.addCleanup(self.pipeline.close)

    def test_motion_without_further_decoding_keeps_axes_and_estimates_new_pose(self):
        p = self.pipeline
        initial = p.process(self.images[0], 0)
        initial_keys = {o.key for o in initial if display_source(o, p.track)}
        self.assertGreaterEqual(len(initial_keys), 68)
        # Force scheduled decoder refreshes to fail. Flow must still measure
        # motion; merely holding the initial pose would fail the accuracy check.
        p.settings["detection_interval_s"] = .04
        with patch.object(p.detector, "observe", return_value=[]) as decoder:
            for i in range(1, len(self.images)):
                obs = p.process(self.images[i], i/30)
                self.assertIsNotNone(p.track.measurement)
                self.assertFalse(p.track.display_predicted)
                self.assertGreater(sum(o.source == "flow" for o in obs), 60)
                visible = {o.key for o in obs if display_source(o, p.track)}
                self.assertEqual(visible, initial_keys)
                self.assertLess(np.linalg.norm(p.track.measurement.pose[:3, 3]-self.truth[i][:3, 3]), .002)
                self.assertLess(np.rad2deg(rotation_distance(p.track.measurement.pose[:3, :3], self.truth[i][:3, :3])), 1)
            self.assertGreater(decoder.call_count, 0)
        self.assertGreater(p.track.measurement.pose[0, 3]-self.truth[0][0, 3], .006)

    def test_board_projection_restores_ids_without_feeding_estimation(self):
        p = self.pipeline
        p.process(self.images[0], 0)
        decoded = p.detector.observe(self.images[1])
        partial = [o for o in decoded if o.marker_id in (7, 28)]
        self.assertEqual(len(partial), 4)
        p.last_decode = None
        with patch.object(p, "_flow_eye", return_value=[]), patch.object(p.detector, "observe", return_value=partial):
            obs = p.process(self.images[1], 1/30)
        self.assertIsNotNone(p.track.measurement)
        self.assertEqual(p.solver.diagnostics["observations"], 4)
        projected = [o for o in obs if o.source == "board"]
        self.assertGreater(len(projected), 50)
        self.assertTrue(all(o.appearance_valid and o.display_source == "board" for o in projected))
        self.assertTrue(all(o.source != "board" for eye in p.previous_obs.values() for o in eye))
        self.assertTrue(all(o.key not in p.track.measurement.inliers for o in projected))

    def test_weak_tag_grace_expires_without_refreshing_its_own_deadline(self):
        p = self.pipeline
        p.process(self.images[0], 0)
        decoded = p.detector.observe(self.images[0])
        partial = [o for o in decoded if o.marker_id in (7, 28)]
        key = ("cam0", 14)
        self.assertIn(key, p.last_verified)
        appearance = p._appearances
        def weaker(observations, gray):
            appearance(observations, gray)
            for obs in observations:
                if obs.key == key:
                    obs.appearance_valid, obs.appearance_weak = False, True
                    obs.appearance_score = .92
        with patch.object(p, "_flow_eye", return_value=[]), \
             patch.object(p.detector, "observe", side_effect=lambda _: [replace(o) for o in partial]), \
             patch.object(p, "_appearances", side_effect=weaker):
            obs = p.process(self.images[0], .06)
            shown = next(o for o in obs if o.key == key)
            self.assertEqual(shown.display_source, "board_grace")
            self.assertAlmostEqual(shown.confirmation_age_s, .06)
            self.assertEqual(p.last_verified[key], 0)
            obs = p.process(self.images[0], .2)
            self.assertFalse(any(o.key == key and display_source(o, p.track) for o in obs))
            self.assertIsNotNone(p.track.measurement)

    def test_blank_images_remove_overlays_and_board_can_reacquire(self):
        p = self.pipeline
        p.process(self.images[0], 0)
        blanks = tuple(np.full_like(image, 160) for image in self.images[0])
        obs = p.process(blanks, 1/30)
        self.assertEqual(obs, [])
        self.assertIsNone(p.track.measurement)
        self.assertIsNone(p.track.display_pose)
        self.assertFalse(p.track.display_predicted)
        obs = p.process(self.images[2], 2/30)
        self.assertIsNotNone(p.track.measurement)
        self.assertGreater(sum(display_source(o, p.track) is not None for o in obs), 60)

    def test_known_pattern_verification_rejects_wrong_identity(self):
        p = self.pipeline
        obs = p.process(self.images[0], 0)
        verified = next(o for o in obs if o.appearance_valid and o.source == "decoded")
        gray = cv2.cvtColor(self.images[0][0 if verified.camera == "cam0" else 1], cv2.COLOR_BGR2GRAY)
        for marker_id in ((verified.marker_id+1) % 36, (verified.marker_id+7) % 36):
            wrong = replace(verified, marker_id=marker_id)
            self.assertFalse(p._appearance(wrong, gray))

    def test_duplicate_id_cannot_survive_as_board_overlay(self):
        p = self.pipeline
        p.process(self.images[0], 0)
        decoded = p.detector.observe(self.images[1])
        key = decoded[0].key
        decoded[0].reason = "duplicate_id"
        p.last_decode = None
        with patch.object(p.detector, "observe", return_value=decoded):
            obs = p.process(self.images[1], 1/30)
        rejected = next(o for o in obs if o.key == key)
        self.assertIsNone(display_source(rejected, p.track))
        self.assertTrue(all(o.key != key for eye in p.previous_obs.values() for o in eye))

    def test_time_gap_forces_reacquisition_instead_of_tracking_old_pixels(self):
        p = self.pipeline
        p.process(self.images[0], 0)
        with patch.object(p, "_flow_eye", wraps=p._flow_eye) as flow, \
             patch.object(p.detector, "observe", wraps=p.detector.observe) as decoder:
            p.process(self.images[2], 1.)
        flow.assert_not_called()
        decoder.assert_called_once()
        self.assertIsNotNone(p.track.measurement)
        # Media-clock rewind also resets motion prediction and reacquires.
        p.process(self.images[0], 0.)
        self.assertIsNotNone(p.track.measurement)
        np.testing.assert_array_equal(p.track.velocity, np.zeros(3))

    def test_active_viewer_batches_are_replaced_without_clear_between_frames(self):
        p = self.pipeline
        obs = p.process(self.images[0], 0)
        viewer = Visualizer.__new__(Visualizer)
        viewer.cfg, viewer.cameras, viewer.board = self.cfg, self.cameras, self.board
        viewer.transforms, viewer.recording = p.solver.transforms, MagicMock()
        viewer.log(Pair(self.images[0], 0, 0, 1), 0, obs, p.track, 1, p.solver.diagnostics)
        paths = [call.args[0] for call in viewer.recording.log.call_args_list]
        for name in self.cameras:
            self.assertNotIn(f"/tracking/{name}/overlays", paths)
            self.assertIn(f"/tracking/{name}/overlays/axes", paths)

    def test_recorded_sequence_preserves_current_image_support(self):
        # Recorded pixels must use their original calibration, not the current
        # device profile, which can legitimately be recalibrated later.
        p = Pipeline(self.cfg, load_calibration(FIXTURES/"calibration.yaml"),
                     Board(read_yaml(FIXTURES/"target.yaml")))
        self.addCleanup(p.close)
        manifest = json.loads((FIXTURES/"manifest.json").read_text())
        sequence = manifest["temporal_sequence"]
        counts = []
        for frame in sequence:
            images = tuple(cv2.imread(str(FIXTURES/f"{frame['frame']}-{name}.jpg")) for name in ("cam0", "cam1"))
            obs = p.process(images, frame["time_s"]-sequence[0]["time_s"])
            self.assertIsNotNone(p.track.measurement, p.solver.diagnostics)
            counts.append(sum(o.source == "flow" for o in obs))
            for o in obs:
                if o.source == "flow" or (o.source == "board" and o.display_source != "board_grace"):
                    self.assertTrue(o.appearance_valid)
                if o.display_source == "board_grace":
                    self.assertLessEqual(o.confirmation_age_s, p.settings["tag_grace_s"])
                if o.key in p.track.measurement.inliers:
                    self.assertLessEqual(o.reprojection_px, 2.5)
        self.assertGreater(sum(counts[1:]), 0)


class PredictionTests(unittest.TestCase):
    def test_prediction_is_bounded_moves_and_does_not_become_a_measurement(self):
        cfg, _, _ = load_config()
        track = Track(cfg["tracking"])
        pose = np.eye(4)
        pose[:3, :3] = np.diag([1., -1., -1.])
        pose[2, 3] = .5
        track.update(0, Measurement(pose.copy(), 0, "stereo", frozenset(), {}), "ok")
        pose[0, 3] = .01
        track.update(.03, Measurement(pose.copy(), 0, "stereo", frozenset(), {}), "ok")
        track.update(.06, None, "insufficient_inliers", image_support=True)
        self.assertIsNone(track.measurement)
        self.assertTrue(track.display_predicted)
        self.assertGreater(track.display_pose[0, 3], .01)
        self.assertEqual(track.last_stamp, .03)
        track.update(.19, None, "insufficient_inliers", image_support=True)
        self.assertIsNone(track.display_pose)
        track.update(.08, None, "ambiguous_pose", image_support=True)
        self.assertIsNone(track.display_pose)

    def test_predicted_display_is_separate_in_export(self):
        cfg, _, _ = load_config()
        cameras = load_calibration(FIXTURES/"calibration.yaml")
        board = Board(read_yaml(FIXTURES/"target.yaml"))
        p = Pipeline(cfg, cameras, board)
        try:
            images = tuple(cv2.imread(str(FIXTURES/f"548-{name}.jpg")) for name in ("cam0", "cam1"))
            obs = p.process(images, 0)
            p.track.update(.03, None, "insufficient_inliers", image_support=True)
            for o in obs:
                o.display_source = "predicted"
            row = frame_record(Pair(images, .03, 0, 2), .03, obs, p.track, board, p.solver, 1)
            self.assertIsNone(row["T_reference_board"])
            self.assertIsNotNone(row["display_T_reference_board"])
            self.assertTrue(row["display_predicted"])
            self.assertTrue(all(not t["valid_pose"] and "T_camera_tag" not in t for t in row["tags"]))
            self.assertTrue(all(t["display_pose_source"] == "predicted" and "T_display_camera_tag" in t for t in row["tags"]))
        finally:
            p.close()


if __name__ == "__main__":
    unittest.main()
