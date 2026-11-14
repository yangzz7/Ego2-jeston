"""Regression checks for fisheye pixel alignment, temporal state and stereo pairing."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from ego2.common.config import camera, orient
from ego2.common.geometry import rotation
from ego2.common.sources import Pair
from ego2.gesture.app import FrameClock, LatestPairs, Pipeline
from ego2.gesture.config import DEFAULTS, load_config
from ego2.gesture.recognizer import Detection
from ego2.gesture.rectification import Rectifier, undo_orientation
from ego2.gesture.tracking import HandTracker


def test_camera():
    block = dict(camera_model="pinhole", distortion_model="equidistant",
                 intrinsics=[180, 183, 319.5, 239.5],
                 distortion_coeffs=[0.01, -0.002, 0, 0], resolution=[640, 480])
    return camera("cam0", block, np.eye(4), "kalibr")


def detection(x=0.5, handedness="Left", pinch=0.1, gesture="Open_Palm", score=0.9):
    points = np.full((21, 3), [x, 0.5, 0.0], dtype=np.float32)
    points[0, 1] = 0.7
    points[9, 1] = 0.4
    points[5, 0] -= 0.1
    points[17, 0] += 0.1
    points[4, :2] = [x, 0.3]
    points[8, :2] = [x + pinch * 0.3, 0.3]
    return Detection(points, handedness, 0.95, gesture, score)


class ProjectionTests(unittest.TestCase):
    def test_inverse_orientation_for_all_rotations_and_flips(self):
        # Pixel IDs constitute an independent oracle for the coordinate inverse.
        raw = np.arange(7 * 11).reshape(7, 11)
        for angle in (0, 90, 180, 270):
            for fh in (False, True):
                for fv in (False, True):
                    channel = dict(rotate_cw=angle, flip_horizontal=fh, flip_vertical=fv)
                    changed = orient(raw, channel)
                    yy, xx = np.indices(changed.shape)
                    inverse = undo_orientation(np.column_stack((xx.ravel(), yy.ravel())), (11, 7), channel).astype(int)
                    np.testing.assert_array_equal(raw[inverse[:, 1], inverse[:, 0]], changed.ravel())

    def test_backprojection_agrees_with_opencv_and_crop_resize(self):
        cam = test_camera()
        settings = dict(width=320, height=240, hfov=75.0, yaw=12.0, pitch=-9.0)
        channel = dict(camera="cam0", calibration_crop=[0, 40, 640, 400])
        rectifier = Rectifier(cam, channel, settings)
        rectifier.prepare(np.zeros((200, 320, 3), np.uint8))
        uv = np.array([[80, 70], [160, 120], [240, 170]], np.float32)
        points, valid = rectifier.backproject(uv / [320, 240])
        focal = 320 / (2 * np.tan(np.deg2rad(75 / 2)))
        rays = np.column_stack(((uv[:, 0] - 159.5) / focal,
                                (uv[:, 1] - 119.5) / focal, np.ones(3)))
        rays = rays @ rotation(12, -9).T
        expected, _ = cv2.fisheye.projectPoints(rays.reshape(1, -1, 3), np.zeros(3), np.zeros(3), cam.K, cam.D)
        expected = (expected.reshape(-1, 2) - [0, 40] + 0.5) * 0.5 - 0.5
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points, expected, atol=0.03)

    def test_raw_input_is_unchanged_and_invalid_points_are_masked(self):
        cam = test_camera()
        rectifier = Rectifier(cam, dict(camera="cam0"), dict(width=320, height=320, hfov=100, pitch=0))
        raw = np.random.default_rng(9).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        before = raw.copy()
        rectifier.prepare(raw)
        np.testing.assert_array_equal(raw, before)
        pixels, valid = rectifier.backproject(np.array([[0.5, 0.5], [-0.1, 0.5], [1, 0.5], [np.nan, 0.5]]))
        np.testing.assert_array_equal(valid, [True, False, False, False])
        self.assertTrue(np.isnan(pixels[1:]).all())

    def test_monocular_projection_does_not_depend_on_stereo_translation(self):
        cam = test_camera()
        other = replace(cam, T_from_root=np.eye(4))
        other.T_from_root[:3, 3] = [1.0, 0.2, -0.5]
        kwargs = dict(width=320, height=320, hfov=100, pitch=0)
        maps = []
        for current in (cam, other):
            rectifier = Rectifier(current, dict(camera="cam0"), kwargs)
            rectifier.prepare(np.zeros((480, 640, 3), np.uint8))
            maps.append(rectifier.maps.xy[0])
        np.testing.assert_array_equal(maps[0], maps[1])


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = HandTracker(DEFAULTS)

    def test_reordered_detections_keep_individual_ids(self):
        left, right = detection(0.25), detection(0.75, "Right")
        first = self.tracker.update([left, right], 0)
        second = self.tracker.update([right, left], 0.033)
        self.assertEqual([h.id for h in second], [first[1].id, first[0].id])

    def test_missing_hands_not_drawn_and_long_gap_gets_new_id(self):
        first = self.tracker.update([detection()], 0)[0]
        self.assertEqual(self.tracker.update([], 0.05), [])
        short = self.tracker.update([detection()], 0.1)[0]
        self.assertEqual(first.id, short.id)
        long = self.tracker.update([detection()], 0.6)[0]
        self.assertNotEqual(first.id, long.id)
        self.assertEqual(long.gesture, "Unknown")

    def test_occlusion_cannot_complete_a_pending_pinch(self):
        self.tracker.update([detection(pinch=0.1)], 0)
        self.tracker.update([], 0.08)
        hand = self.tracker.update([detection(pinch=0.1)], 0.15)[0]
        self.assertEqual(hand.pinch, "Open")
        self.assertEqual(hand.gesture, "Unknown")

    def test_label_confirmation_and_unknown_clear(self):
        self.assertEqual(self.tracker.update([detection()], 0)[0].gesture, "Unknown")
        self.assertEqual(self.tracker.update([detection()], 0.13)[0].gesture, "Open_Palm")
        pending = self.tracker.update([detection(gesture="Closed_Fist")], 0.16)[0]
        self.assertEqual(pending.gesture, "Open_Palm")
        self.assertIsNone(pending.score)  # Never attach the new class's score to the old class.
        low = self.tracker.update([detection(score=0.2)], 0.19)[0]
        self.assertEqual(low.gesture, "Unknown")
        self.assertIsNone(low.score)

    def test_pinch_hysteresis_release_and_visibility(self):
        self.assertEqual(self.tracker.update([detection(pinch=0.1)], 0)[0].pinch, "Open")
        self.assertEqual(self.tracker.update([detection(pinch=0.1)], 0.13)[0].pinch, "Pinching")
        self.assertEqual(self.tracker.update([detection(pinch=0.32)], 0.17)[0].pinch, "Pinching")
        self.tracker.update([detection(pinch=0.5)], 0.2)
        self.assertEqual(self.tracker.update([detection(pinch=0.5)], 0.33)[0].pinch, "Released")
        self.assertEqual(self.tracker.update([detection(pinch=0.5)], 0.59)[0].pinch, "Open")
        mask = np.ones(21, bool)
        mask[8] = False
        self.assertEqual(self.tracker.update([detection()], 0.62, [mask])[0].pinch, "Unknown")

    def test_jitter_filter_moves_toward_current_observation(self):
        initial = self.tracker.update([detection(0.4)], 0)[0]
        moved = self.tracker.update([detection(0.45)], 0.03)[0]
        self.assertEqual(initial.id, moved.id)
        self.assertGreater(moved.points[0, 0], 0.4)
        self.assertLess(moved.points[0, 0], 0.45)


class RuntimeTests(unittest.TestCase):
    def test_source_time_is_preserved_and_resets_are_rejected(self):
        clock = FrameClock()
        self.assertEqual(clock.update(1234.0), (0.0, 0))
        seconds, ms = clock.update(1234.0001)
        self.assertAlmostEqual(seconds, 0.0001)
        self.assertEqual(ms, 1)
        with self.assertRaisesRegex(ValueError, "backwards"):
            clock.update(1234.0)

    def test_latest_slot_drops_whole_pairs_and_closes_source(self):
        class FakeSource:
            def __init__(self):
                self.count = 0
                self.closed = False
            def open(self):
                pass
            def read(self, stop):
                self.count += 1
                if self.count > 20:
                    return None
                return Pair((self.count, -self.count), self.count, 0, self.count)
            def close(self):
                self.closed = True
        source = FakeSource()
        capture = LatestPairs(source, threading.Event()).start()
        capture.thread.join(timeout=2)
        pair = capture.read()
        self.assertEqual(pair.images, (20, -20))
        self.assertEqual(capture.dropped, 19)
        self.assertIsNone(capture.read())
        capture.close()
        self.assertTrue(source.closed)

    def test_capture_errors_reach_caller(self):
        source = SimpleNamespace(open=lambda: None, close=lambda: None,
                                 read=lambda stop: (_ for _ in ()).throw(ValueError("broken input")))
        capture = LatestPairs(source, threading.Event()).start()
        with self.assertRaisesRegex(RuntimeError, "broken input"):
            capture.read()
        capture.close()

    def test_two_independent_models_receive_same_time_and_return_raw_coordinates(self):
        class FakeRecognizer:
            calls = []
            def __init__(self, settings):
                self.times = []
                self.calls.append(self)
            def detect(self, rgb, timestamp):
                self.times.append(timestamp)
                return [replace(detection(), world_points=np.full((21, 3), 0.123, np.float32),
                                gesture_categories=[dict(category_name="Open_Palm", score=0.9)])]
            def close(self):
                pass
        profile, cameras = load_config()
        pair = Pair(tuple(np.zeros((1024, 1280, 3), np.uint8) for _ in range(2)), 99.0, 0, 1)
        with patch("ego2.gesture.app.Recognizer", FakeRecognizer):
            pipeline = Pipeline(profile, cameras)
            stamp, hands = pipeline.process(pair)
            self.assertEqual(stamp, 0)
            self.assertEqual(len(FakeRecognizer.calls), 2)
            self.assertEqual(FakeRecognizer.calls[0].times, FakeRecognizer.calls[1].times)
            self.assertEqual([len(h) for h in hands], [1, 1])
            self.assertGreater(hands[0][0].points[0, 0], 100)
            self.assertFalse(np.allclose(hands[0][0].points, hands[1][0].points))
            raw = pipeline.observations[0][0]
            self.assertTrue(raw["accepted"])
            self.assertEqual(raw["track_id"], hands[0][0].id)
            self.assertEqual(raw["gesture_categories"][0]["category_name"], "Open_Palm")
            np.testing.assert_allclose(raw["world_points"], 0.123)
            self.assertEqual(raw["points"].shape, (21, 3))
            self.assertEqual(raw["smoothed_points"].shape, (21, 3))
            pipeline.close()

    def test_offline_video_processes_every_pair_and_stops_at_eof(self):
        from ego2.gesture.__main__ import main
        class FakeRecognizer:
            def __init__(self, settings):
                pass
            def detect(self, rgb, timestamp):
                return []
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            video = str(Path(tmp) / "stereo.avi")
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 256))
            self.assertTrue(writer.isOpened())
            for i in range(3):
                writer.write(np.full((256, 640, 3), i * 50, np.uint8))
            writer.release()
            metrics = Path(tmp) / "metrics.json"
            dataset = Path(tmp) / "dataset"
            with patch("ego2.gesture.app.Recognizer", FakeRecognizer):
                self.assertEqual(main(["--video", video, "--no-viz", "--metrics", str(metrics),
                                       "--dataset", str(dataset)]), 0)
            import json
            self.assertEqual(json.loads(metrics.read_text())["processed_pairs"], 3)
            self.assertEqual(json.loads((dataset / "meta/info.json").read_text())["fps"], 10)
            self.assertEqual(json.loads(metrics.read_text())["captured_pairs"], 3)


class VisualizationTests(unittest.TestCase):
    def test_compressed_rerun_preview_keeps_both_image_frames(self):
        from ego2.gesture.visualizer import Visualizer
        import rerun as rr
        viewer = Visualizer.__new__(Visualizer)
        viewer.jpeg_quality = 95
        viewer.recording = MagicMock()
        pair = Pair(tuple(np.full((20, 20, 3), 120, np.uint8) for _ in range(2)), 1, 0, 7)
        viewer.log(pair, 0, [[], []])
        images = [call.args[1] for call in viewer.recording.log.call_args_list if call.args[0].endswith('/rgb')]
        self.assertEqual(len(images), 2)
        self.assertTrue(all(isinstance(value, rr.EncodedImage) for value in images))

    def test_native_save_and_display_share_explicit_recording(self):
        from ego2.gesture.visualizer import Visualizer
        profile, _ = load_config()
        recording = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("ego2.common.viewer.sys.platform", "darwin"), \
                patch("ego2.gesture.visualizer.rr.RecordingStream", return_value=recording), \
                patch("ego2.gesture.visualizer.rr.spawn", side_effect=AssertionError("Global recording is disabled")):
            viewer = Visualizer(profile, save=Path(tmp) / "hands.rrd")
            self.assertEqual(recording.spawn.call_count, 1)
            self.assertFalse(recording.spawn.call_args.kwargs["connect"])
            self.assertEqual(len(recording.set_sinks.call_args.args), 2)
            viewer.close()

    def test_failed_setup_stops_owned_web_server(self):
        from ego2.gesture.visualizer import Visualizer
        profile, _ = load_config()
        process = MagicMock()
        def start(viewer, *args):
            viewer.web_process = process
            return "rerun+http://127.0.0.1:9876/proxy"
        recording = MagicMock()
        recording.connect_grpc.side_effect = RuntimeError("connection failed")
        with patch("ego2.gesture.visualizer.rr.RecordingStream", return_value=recording), \
                patch.object(Visualizer, "_start_web", start):
            with self.assertRaisesRegex(RuntimeError, "connection failed"):
                Visualizer(profile, web=True)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()

    def test_overlay_streams_are_excluded_from_raw_views(self):
        from ego2.gesture.visualizer import build_blueprint
        profile, _ = load_config()
        blueprint = build_blueprint(profile["channels"])
        views = [view for row in blueprint.root_container.contents for view in row.contents]
        self.assertEqual(len(views), 4)
        for i, view in enumerate(views):
            text = str(view.contents)
            if i < 2:
                self.assertNotIn("hands", text)
            else:
                self.assertIn("hands", text)

    def test_each_frame_clears_old_annotations_without_duplicate_rgb(self):
        from ego2.gesture.visualizer import Visualizer
        class Recording:
            def __init__(self):
                self.logs = []
                self.times = []
            def log(self, path, value):
                self.logs.append((path, type(value).__name__))
            def set_time(self, *args, **kwargs):
                self.times.append((args, kwargs))
        viewer = Visualizer.__new__(Visualizer)
        viewer.recording = Recording()
        pair = Pair(tuple(np.zeros((20, 20, 3), np.uint8) for _ in range(2)), 1.0, 0, 7)
        viewer.log(pair, 0.5, [[], []])
        self.assertEqual(sum(kind == "Image" for _, kind in viewer.recording.logs), 2)
        self.assertEqual([path for path, kind in viewer.recording.logs if kind == "Clear"],
                         ["/head/eye0/hands", "/head/eye1/hands"])
        self.assertEqual(viewer.recording.times[0], (("capture_time",), {"duration": 0.5}))


if __name__ == "__main__":
    unittest.main()
