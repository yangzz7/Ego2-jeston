"""Display/measurement separation, slow-ISP frame pairing and overlay alignment."""
import copy
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from ego2.aprilgrid.config import Board, load_config
from ego2.common.config import load_calibration, read_yaml
from ego2.aprilgrid.display_isp import IspVisualizer, display_isp_active
from ego2.aprilgrid.temporal import Pipeline
from ego2.aprilgrid.visualizer import Visualizer
from ego2.common.sources import Pair, decode_cyper, white_balance


class BlockingIsp:
    last_white_balance_rgb = np.ones((2, 3))

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.closed = False

    def process_transport(self, raw):
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("Test did not release ISP")
        return tuple(np.repeat(raw[:, :, i:i+1], 3, axis=2) for i in range(2))

    def close(self):
        self.closed = True


def pair(number):
    images = tuple(np.full((8, 8, 3), number+i, np.uint8) for i in range(2))
    raw = np.full((8, 8, 2), number+20, np.uint8)
    return Pair(images, number/30, 0, number, raw_transport=raw)


class DisplayQueueTests(unittest.TestCase):
    def test_slow_isp_keeps_complete_snapshots_and_all_raw_recordings(self):
        sink, processor = MagicMock(), BlockingIsp()
        worker = IspVisualizer(sink, processor)
        track = SimpleNamespace(display_pose=np.eye(4), status="first")
        observations = [SimpleNamespace(pixels=np.array([[1., 2.]]))]
        try:
            worker.log(pair(1), 1/30, observations, track, 2, {"count": 1}, {"frame": 1})
            self.assertTrue(processor.entered.wait(2))
            track.display_pose[0, 3] = 2
            track.status = "second"
            observations[0].pixels[:] = 2
            worker.log(pair(2), 2/30, observations, track, 2)
            track.display_pose[0, 3] = 3
            observations[0].pixels[:] = 3
            worker.log(pair(3), 3/30, observations, track, 2)
            track.display_pose[:] = 99
            observations[0].pixels[:] = 99
            # All submissions finished while the ISP is still blocked.
            self.assertFalse(processor.release.is_set())
            self.assertEqual(sink.log_raw.call_count, 3)
        finally:
            processor.release.set()
            worker.close()
        frames = sink.log.call_args_list
        self.assertEqual([c.args[0].sequence for c in frames], [1, 3])
        self.assertEqual([c.args[1] for c in frames], [1/30, 3/30])
        for call, number, tx in zip(frames, (1, 3), (0, 3)):
            self.assertEqual(call.args[3].display_pose[0, 3], tx)
            self.assertEqual(call.args[2][0].pixels[0, 0], number)
            self.assertEqual(call.kwargs["display_info"]["frame"], number)
            self.assertTrue(np.all(call.kwargs["display_images"][0] == number+20))
            self.assertTrue(np.all(call.args[0].images[0] == number))
            self.assertFalse(call.kwargs["record_raw"])
        self.assertEqual(worker.statistics()["dropped_display_frames"], 1)
        self.assertEqual(worker.statistics()["enhanced_frames"], 2)
        self.assertEqual(worker.statistics()["last_displayed_frame"], 3)
        self.assertTrue(processor.closed)
        sink.close.assert_called_once()

    def test_missing_raw_error_and_wrong_geometry_use_current_original_pair(self):
        for failure in ("missing", "exception", "geometry"):
            with self.subTest(failure=failure):
                sink, processor = MagicMock(), MagicMock()
                processor.last_white_balance_rgb = np.ones((2, 3))
                if failure == "exception":
                    processor.process_transport.side_effect = RuntimeError("bad metadata")
                else:
                    processor.process_transport.return_value = (np.zeros((2, 2, 3), np.uint8),)*2
                item = pair(17)
                if failure == "missing":
                    item.raw_transport = None
                original = tuple(image.copy() for image in item.images)
                worker = IspVisualizer(sink, processor, record_raw=False)
                worker.log(item, .7, [], SimpleNamespace(status="not_seen"), 1)
                worker.close()
                call = sink.log.call_args
                self.assertIs(call.kwargs["display_images"], item.images)
                self.assertEqual(call.kwargs["display_info"]["status"], "fallback")
                self.assertEqual(call.args[0].sequence, 17)
                for before, after in zip(original, item.images):
                    np.testing.assert_array_equal(before, after)
                sink.log_raw.assert_not_called()

    def test_sink_error_is_reported_and_resources_close(self):
        sink, processor = MagicMock(), BlockingIsp()
        processor.release.set()
        sink.log.side_effect = RuntimeError("sink failed")
        worker = IspVisualizer(sink, processor, record_raw=False)
        worker.log(pair(1), 0, [], None, 1)
        with self.assertRaisesRegex(RuntimeError, "sink failed"):
            worker.close()
        self.assertTrue(processor.closed)
        sink.close.assert_called_once()


class DisplayImageTests(unittest.TestCase):
    def test_four_views_change_pixels_but_preserve_raw_images_axes_ids_and_poses(self):
        cfg, cameras, board = load_config()
        cv2.setNumThreads(1)
        fixtures = Path(__file__).parent/"fixtures"/"aprilgrid"
        # These recorded images belong to the bundled calibration snapshot,
        # not to whichever s200032 calibration the user installed most recently.
        cameras = load_calibration(fixtures/"calibration.yaml")
        board = Board(read_yaml(fixtures/"target.yaml"))
        images = tuple(cv2.imread(str(fixtures/f"972-{name}.jpg")) for name in ("cam0", "cam1"))
        originals = tuple(image.copy() for image in images)
        pipeline = Pipeline(cfg, cameras, board)
        try:
            obs = pipeline.process(images, 0)
            self.assertIsNotNone(pipeline.track.measurement)
            pose_before = pipeline.track.measurement.pose.copy()
            viewer = Visualizer.__new__(Visualizer)
            viewer.cfg, viewer.cameras, viewer.board = cfg, cameras, board
            viewer.transforms, viewer.recording = pipeline.solver.transforms, MagicMock()
            sample = Pair(images, 0, 0, 972)
            viewer.log(sample, 0, obs, pipeline.track, 10)
            baseline = {c.args[0]: c.args[1] for c in viewer.recording.log.call_args_list}
            viewer.recording.reset_mock()
            enhanced = tuple(np.full_like(image, [20, 70, 140]) for image in images)
            viewer.log(sample, 0, obs, pipeline.track, 10, display_images=enhanced,
                       display_info={"status": "enhanced", "processing_ms": 40, "frame": 972})
            actual = {c.args[0]: c.args[1] for c in viewer.recording.log.call_args_list}
            for name in cameras:
                top, bottom = actual[f"/display/{name}/image"], actual[f"/tracking/{name}/image"]
                self.assertIs(top, bottom)
                blob = top.blob.as_arrow_array().to_pylist()[0]
                decoded = cv2.imdecode(np.frombuffer(bytes(blob), np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_allclose(decoded[100, 100], [20, 70, 140], atol=2)
                raw = f"/raw/{name}/image"
                self.assertTrue(actual[raw].blob.as_arrow_array().equals(baseline[raw].blob.as_arrow_array()))
                for entity, field in (("outlines", "strips"), ("axes", "strips"), ("ids", "labels"), ("ids", "positions")):
                    path = f"/tracking/{name}/overlays/{entity}"
                    self.assertTrue(getattr(actual[path], field).as_arrow_array().equals(
                        getattr(baseline[path], field).as_arrow_array()), path)
                readout = f"/camera_pose/{name}/readout"
                self.assertTrue(actual[readout].text.as_arrow_array().equals(baseline[readout].text.as_arrow_array()))
            for before, after in zip(originals, images):
                np.testing.assert_array_equal(before, after)
            np.testing.assert_array_equal(pipeline.track.measurement.pose, pose_before)
        finally:
            pipeline.close()

    def test_display_isp_requires_native_source_and_visualization(self):
        cfg, _, _ = load_config()
        args = SimpleNamespace(no_viz=False, demo=False)
        self.assertTrue(display_isp_active(cfg, args))
        args.no_viz = True
        self.assertFalse(display_isp_active(cfg, args))
        args.no_viz, args.demo = False, True
        self.assertFalse(display_isp_active(cfg, args))
        args.demo = False
        for kind in ("video", "images", "ros2"):
            cfg["input"]["type"] = kind
            self.assertFalse(display_isp_active(cfg, args))

    def test_native_isp_preserves_transport_and_detector_images(self):
        cfg, _, _ = load_config()
        settings = cfg["viewer"]["display_isp"]
        if not Path(settings["library"]).exists():
            self.skipTest("Build the optional native ISP library to run this integration test")
        from cyperstereo_isp_portable.python.cyperstereo_isp import Isp
        # Synthetic full-resolution transport with valid hw2/fw6 metadata.
        raw = np.full((1024, 1280, 2), 70, np.uint8)
        raw[-1] = 0
        raw[-1, 0, 0], raw[-1, 1, 0] = 2, 6
        saved_raw = raw.copy()
        images, _ = decode_cyper(raw, 1280, 1024)
        images = tuple(white_balance(image) for image in images)
        before = copy.deepcopy(images)
        with Isp(config=settings["config"], library=settings["library"]) as processor:
            displayed = processor.process_transport(raw)
        np.testing.assert_array_equal(raw, saved_raw)
        for original, unchanged, output in zip(before, images, displayed):
            np.testing.assert_array_equal(original, unchanged)
            self.assertEqual(output.shape, original.shape)
            self.assertGreater(np.abs(output.astype(float)-original).mean(), 10)


if __name__ == "__main__":
    unittest.main()
