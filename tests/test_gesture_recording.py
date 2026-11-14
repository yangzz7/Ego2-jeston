"""Recording lifecycle regressions with real video/Parquet archive finalization."""
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ego2.common.runtime import LatestPairs, QueuedArchive
from ego2.common.sources import Pair
from ego2.gesture.app import run
from ego2.gesture.dataset import GestureDataset
from ego2.gesture.replay import DatasetReplay


AVAILABLE = all(importlib.util.find_spec(name) for name in ("pyarrow", "pandas")) and bool(shutil.which("ffmpeg"))


def pair(sequence):
    image = np.full((18, 24, 3), sequence * 20, np.uint8)
    return Pair((image, image.copy()), 10 + sequence / 10, time.monotonic(), sequence)


def profile(kind="cyper"):
    return {"input": {"type": kind}, "channels": [{"camera": "cam0"}, {"camera": "cam1"}]}


def arguments(root, **overrides):
    values = dict(cameras={}, dataset=root, dataset_fps=10, task="lifecycle test", episode_frames=2,
                  no_viz=True, headless=True, save=None, connect=None, web=False,
                  web_port=9090, grpc_port=9876, frames=0, seconds=0,
                  metrics=root / "metrics.json")
    values.update(overrides)
    return SimpleNamespace(**values)


@unittest.skipUnless(AVAILABLE, "recording needs pyarrow, pandas and ffmpeg")
class RecordingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"

    def test_successful_live_capture_finalizes_all_frames_and_sparse_inference(self):
        inference_started, capture_finished = threading.Event(), threading.Event()
        source_closed, pipeline_closed = threading.Event(), threading.Event()
        processed = []

        class FakeSource:
            kind, fps = "cyper", 30

            def __init__(self, *args):
                self.sequence = 0

            def open(self):
                self.fps = 10  # Device negotiation differs from requested FPS.

            def read(self, stop):
                if self.sequence == 6:
                    capture_finished.set()
                    return None
                if self.sequence and not inference_started.wait(5):
                    raise TimeoutError("Test inference did not start")
                self.sequence += 1
                return pair(self.sequence)

            def close(self):
                source_closed.set()

        class FakePipeline:
            observations = [[], []]

            def __init__(self, *args):
                pass

            def process(self, current):
                processed.append(current.sequence)
                inference_started.set()
                if not capture_finished.wait(5):
                    raise TimeoutError("Test capture did not finish")
                return (current.sequence - 1) / 10, [[], []]

            def close(self):
                pipeline_closed.set()

        class DelayedDataset(GestureDataset):
            def capture(self, current):
                # Simulate an encoder flush holding the archive worker. The
                # camera must continue reading while disk writes are blocked.
                if current.sequence == 1 and not capture_finished.wait(5):
                    raise TimeoutError("Archive blocked the camera reader")
                super().capture(current)

        with patch("ego2.gesture.app.Source", FakeSource), \
             patch("ego2.gesture.app.Pipeline", FakePipeline), \
             patch("ego2.gesture.dataset.GestureDataset", DelayedDataset):
            metrics = run(profile(), arguments(self.root, dataset_fps=None))
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "complete")
        self.assertEqual(json.loads((self.root / "meta/info.json").read_text())["fps"], 10)
        self.assertNotIn("error", extra)
        self.assertEqual((extra["capture_count"], extra["inference_count"]), (6, 2))
        self.assertEqual(processed, [1, 6])
        self.assertEqual((metrics["captured_pairs"], metrics["processed_pairs"], metrics["uninferred_pairs"]), (6, 2, 4))
        self.assertGreater(metrics["archive_peak_pending"], 0)
        self.assertLessEqual(metrics["archive_peak_pending"], 32)
        self.assertTrue(source_closed.is_set())
        self.assertTrue(pipeline_closed.is_set())
        restored = list(DatasetReplay(self.root).frames())
        self.assertEqual([item[0].sequence for item in restored], list(range(1, 7)))
        self.assertEqual([item[3]["observation.inference_valid"] for item in restored], [True, False, False, False, False, True])

    def test_frame_returned_during_shutdown_is_archived_before_source_cleanup(self):
        entered, release, source_closed = threading.Event(), threading.Event(), threading.Event()
        stop = threading.Event()
        writer = GestureDataset(self.root, profile(), {}, fps=10)

        class InflightSource:
            def open(self):
                pass

            def read(self, requested_stop):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Test read was not released")
                return pair(1)

            def close(self):
                source_closed.set()

        capture = LatestPairs(InflightSource(), stop, on_pair=writer.capture).start()
        try:
            self.assertTrue(entered.wait(5))
            # Simulate a driver returning the pair it already acquired after
            # the user requests shutdown, while its blocking read was active.
            stop.set()
            release.set()
            capture.close()
        finally:
            stop.set()
            release.set()
            capture.thread.join(10)
            writer.close()
        self.assertTrue(source_closed.is_set())
        self.assertFalse(capture.thread.is_alive())
        self.assertEqual(writer.capture_count, 1)
        restored = list(DatasetReplay(self.root).frames())
        self.assertEqual([item[0].sequence for item in restored], [1])
        self.assertFalse(restored[0][3]["observation.inference_valid"])

    def test_archive_failure_publishes_saved_prefix_despite_later_inference(self):
        inferred = {1: threading.Event(), 2: threading.Event()}
        archive_failed = threading.Event()
        closed = []

        class FakeSource:
            kind, fps = "cyper", 10

            def __init__(self, *args):
                self.sequence = 0

            def open(self):
                pass

            def read(self, stop):
                if self.sequence == 2:
                    if not archive_failed.wait(5):
                        raise TimeoutError("Archive failure did not occur")
                    return None
                if self.sequence and not inferred[self.sequence].wait(5):
                    raise TimeoutError("Inference did not complete")
                self.sequence += 1
                return pair(self.sequence)

            def close(self):
                closed.append("source")

        class FailingArchive(GestureDataset):
            def capture(self, current):
                if current.sequence == 2:
                    try:
                        if not inferred[2].wait(5):
                            raise TimeoutError("Inference did not get ahead of the archive")
                        raise OSError("archive disk full")
                    finally:
                        archive_failed.set()
                super().capture(current)

            def result(self, current, *args):
                super().result(current, *args)
                inferred[current.sequence].set()

        pipeline = SimpleNamespace(observations=[[], []], close=Mock(),
                                   process=lambda current: ((current.sequence - 1) / 10, [[], []]))
        with patch("ego2.gesture.app.Source", FakeSource), \
             patch("ego2.gesture.app.Pipeline", return_value=pipeline), \
             patch("ego2.gesture.dataset.GestureDataset", FailingArchive):
            with self.assertRaisesRegex(RuntimeError, "archive disk full"):
                run(profile(), arguments(self.root))
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "failed")
        self.assertIn("archive disk full", extra["error"])
        self.assertEqual((extra["capture_count"], extra["inference_count"], extra["finalized_frames"]), (1, 2, 1))
        frames = list(DatasetReplay(self.root).frames())
        self.assertEqual([frame[0].sequence for frame in frames], [1])
        self.assertTrue(frames[0][3]["observation.inference_valid"])
        self.assertEqual(len((self.root / "raw/inference.jsonl").read_text().splitlines()), 2)
        self.assertEqual(closed, ["source"])
        pipeline.close.assert_called_once()

    def test_archive_overflow_stops_capture_and_finalizes_all_accepted_pairs(self):
        busy, release = threading.Event(), threading.Event()
        inferred = {1: threading.Event(), 2: threading.Event()}
        queues = []

        class FakeSource:
            kind, fps = "cyper", 10

            def __init__(self, *args):
                self.sequence = 0

            def open(self):
                pass

            def read(self, stop):
                if self.sequence:
                    if not busy.wait(5) or not inferred[self.sequence].wait(5):
                        raise TimeoutError("Archive/inference did not reach the overflow barrier")
                self.sequence += 1
                return pair(self.sequence)

            def close(self):
                release.set()

        class DelayedArchive(GestureDataset):
            def capture(self, current):
                if current.sequence == 1:
                    busy.set()
                    if not release.wait(5):
                        raise TimeoutError("Source did not stop after overflow")
                super().capture(current)

            def result(self, current, *args):
                super().result(current, *args)
                inferred[current.sequence].set()

        def small_queue(callback):
            writer = QueuedArchive(callback, max_pending=1)
            queues.append(writer)
            return writer

        pipeline = SimpleNamespace(observations=[[], []], close=Mock(),
                                   process=lambda current: ((current.sequence - 1) / 10, [[], []]))
        with patch("ego2.gesture.app.Source", FakeSource), \
             patch("ego2.gesture.app.Pipeline", return_value=pipeline), \
             patch("ego2.gesture.dataset.GestureDataset", DelayedArchive), \
             patch("ego2.gesture.app.QueuedArchive", side_effect=small_queue):
            with self.assertRaisesRegex(RuntimeError, "queue is full"):
                run(profile(), arguments(self.root))
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "failed")
        self.assertIn("queue is full", extra["error"])
        self.assertEqual((extra["capture_count"], extra["inference_count"], extra["finalized_frames"]), (2, 2, 2))
        frames = list(DatasetReplay(self.root).frames())
        self.assertEqual([frame[0].sequence for frame in frames], [1, 2])
        self.assertTrue(all(frame[3]["observation.inference_valid"] for frame in frames))
        self.assertFalse(queues[0].thread.is_alive())
        pipeline.close.assert_called_once()

    def test_interrupted_auto_export_preserves_complete_destination_and_dataset(self):
        destination = Path(self.temp.name) / "complete.rrd"
        complete_bytes = b"complete live recording of all three frames"
        source_closed, pipeline_closed, viewer_closed = [], [], []
        replay_calls = []

        class FakeSource:
            kind, fps = "demo", 10

            def __init__(self, *args):
                self.sequence = 0

            def open(self):
                pass

            def read(self, stop):
                if self.sequence == 3:
                    return None
                self.sequence += 1
                return pair(self.sequence)

            def close(self):
                source_closed.append(True)

        class FakePipeline:
            observations = [[], []]

            def __init__(self, *args):
                pass

            def process(self, current):
                return (current.sequence - 1) / 10, [[], []]

            def close(self):
                pipeline_closed.append(True)

        class FakeViewer:
            def __init__(self, *args, save, **kwargs):
                self.path = Path(save)
                self.count = 0

            def log_session(self, *args):
                pass

            def log_frame_metadata(self, *args):
                pass

            def log(self, *args):
                self.count += 1

            def close(self):
                if self.count != 3:
                    raise AssertionError("Expected all three frames in the initial RRD")
                self.path.write_bytes(complete_bytes)
                viewer_closed.append(True)

        def interrupted_export(args):
            replay_calls.append(args)
            Path(args.save).write_bytes(b"incomplete regenerated RRD")
            # Replay handles Ctrl+C by returning the prefix it managed to log.
            return {"replayed_pairs": 1, "inference_pairs": 1, "missing_inference_pairs": 0}

        with patch("ego2.gesture.app.Source", FakeSource), \
             patch("ego2.gesture.app.Pipeline", FakePipeline), \
             patch("ego2.gesture.visualizer.Visualizer", FakeViewer), \
             patch("ego2.gesture.replay.replay", side_effect=interrupted_export):
            with self.assertRaises(RuntimeError):
                run(profile("demo"), arguments(self.root, no_viz=False, save=destination))
        self.assertEqual(destination.read_bytes(), complete_bytes)
        self.assertEqual((len(source_closed), len(pipeline_closed), len(viewer_closed)), (1, 1, 1))
        self.assertEqual(len(replay_calls), 1)
        self.assertNotEqual(Path(replay_calls[0].save), destination)
        self.assertFalse(Path(replay_calls[0].save).exists())
        self.assertEqual(json.loads((self.root / "meta/ego2.json").read_text())["recording_status"], "complete")
        self.assertEqual(len(list(DatasetReplay(self.root).frames())), 3)

    def test_viewer_failures_preserve_capture_inference_finalization_and_full_export(self):
        # A network failure may first surface at any Rerun API call, and a
        # disconnected stream can fail again when ExitStack flushes/closes it.
        for failure_stage in ("startup", "session", "log", "metadata", "close"):
            with self.subTest(failure_stage=failure_stage):
                root = self.root / failure_stage
                destination = root / "preview/viewer.rrd"
                closed, replay_calls = [], []

                class FakeSource:
                    kind, fps = "demo", 10

                    def __init__(self, *args):
                        self.sequence = 0

                    def open(self):
                        pass

                    def read(self, stop):
                        if self.sequence == 3:
                            return None
                        self.sequence += 1
                        return pair(self.sequence)

                    def close(self):
                        closed.append("source")

                class FakePipeline:
                    observations = [[], []]

                    def __init__(self, *args):
                        pass

                    def process(self, current):
                        return (current.sequence - 1) / 10, [[], []]

                    def close(self):
                        closed.append("pipeline")

                class FailingViewer:
                    def __init__(self, *args, **kwargs):
                        if failure_stage == "startup":
                            raise RuntimeError("gRPC startup disconnected")

                    def log_session(self, *args):
                        if failure_stage == "session":
                            raise RuntimeError("gRPC session disconnected")

                    def log(self, *args):
                        if failure_stage == "log":
                            raise RuntimeError("gRPC log disconnected")

                    def log_frame_metadata(self, *args):
                        if failure_stage == "metadata":
                            raise RuntimeError("gRPC metadata disconnected")

                    def close(self):
                        closed.append("viewer")
                        raise RuntimeError("gRPC close disconnected")

                def complete_export(args):
                    replay_calls.append(args)
                    self.assertTrue(args.headless)
                    self.assertIsNone(args.connect)
                    self.assertFalse(args.web)
                    frames = list(DatasetReplay(args.replay).frames())
                    self.assertEqual([frame[0].sequence for frame in frames], [1, 2, 3])
                    self.assertTrue(all(frame[3]["observation.inference_valid"] for frame in frames))
                    self.assertEqual(json.loads((root / "meta/ego2.json").read_text())["recording_status"], "complete")
                    Path(args.save).write_bytes(b"complete offline export")
                    return {"replayed_pairs": len(frames)}

                with patch("ego2.gesture.app.Source", FakeSource), \
                     patch("ego2.gesture.app.Pipeline", FakePipeline), \
                     patch("ego2.gesture.visualizer.Visualizer", FailingViewer), \
                     patch("ego2.gesture.replay.replay", side_effect=complete_export):
                    metrics = run(profile("demo"), arguments(root, no_viz=False, save=destination))
                self.assertEqual((metrics["captured_pairs"], metrics["processed_pairs"], metrics["uninferred_pairs"]), (3, 3, 0))
                extra = json.loads((root / "meta/ego2.json").read_text())
                self.assertNotIn("error", extra)
                expected = ["startup"] if failure_stage == "startup" else ["close"] if failure_stage == "close" else [
                    "startup" if failure_stage == "session" else "log", "close"]
                self.assertEqual([error["stage"] for error in extra["viewer_errors"]], expected)
                self.assertEqual(metrics["viewer_errors"], extra["viewer_errors"])
                self.assertEqual(json.loads((root / "metrics.json").read_text())["viewer_errors"], extra["viewer_errors"])
                self.assertEqual(destination.read_bytes(), b"complete offline export")
                self.assertEqual(len(replay_calls), 1)
                self.assertEqual(closed.count("source"), 1)
                self.assertEqual(closed.count("pipeline"), 1)
                self.assertEqual(closed.count("viewer"), int(failure_stage != "startup"))

    def test_viewer_cleanup_does_not_hide_inference_failure(self):
        source = SimpleNamespace(kind="demo", fps=10, open=Mock(), close=Mock(),
                                 read=Mock(side_effect=[pair(1), pair(2), None]))
        pipeline = SimpleNamespace(observations=[[], []], close=Mock(),
                                   process=Mock(side_effect=[(0, [[], []]), RuntimeError("inference failed")]))
        # This fixture models the standalone viewer, which has no multi-camera
        # log_observations method. An unconstrained Mock invents that method.
        viewer = Mock(spec=['log','close','log_session','log_frame_metadata'],
                      log=Mock(side_effect=RuntimeError("gRPC log disconnected")),
                      close=Mock(side_effect=RuntimeError("gRPC close disconnected")))
        with patch("ego2.gesture.app.Source", return_value=source), \
             patch("ego2.gesture.app.Pipeline", return_value=pipeline), \
             patch("ego2.gesture.visualizer.Visualizer", return_value=viewer):
            with self.assertRaisesRegex(RuntimeError, "^inference failed$"):
                run(profile("demo"), arguments(self.root, no_viz=False))
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "failed")
        self.assertEqual(extra["error"], "inference failed")
        self.assertEqual([error["stage"] for error in extra["viewer_errors"]], ["log", "close"])
        frames = list(DatasetReplay(self.root).frames())
        self.assertEqual([frame[0].sequence for frame in frames], [1, 2])
        self.assertEqual([frame[3]["observation.inference_valid"] for frame in frames], [True, False])
        source.close.assert_called_once()
        pipeline.close.assert_called_once()
        viewer.close.assert_called_once()

    def test_first_interrupt_finishes_inference_commit_and_second_does_not_break_cleanup(self):
        previous = signal.getsignal(signal.SIGINT)
        observed = []

        class InterruptingDataset(GestureDataset):
            def result(self, *args):
                signal.raise_signal(signal.SIGINT)
                super().result(*args)
                observed.append("inference committed")

            def close(self, *args, **kwargs):
                signal.raise_signal(signal.SIGINT)
                super().close(*args, **kwargs)
                observed.append("dataset finalized")

        source = SimpleNamespace(kind="demo", fps=10, open=Mock(), close=Mock(),
                                 read=Mock(side_effect=[pair(1), pair(2), None]))
        pipeline = SimpleNamespace(observations=[[], []], close=Mock(),
                                   process=lambda current: ((current.sequence - 1) / 10, [[], []]))
        with patch("ego2.gesture.app.Source", return_value=source), \
             patch("ego2.gesture.app.Pipeline", return_value=pipeline), \
             patch("ego2.gesture.dataset.GestureDataset", InterruptingDataset):
            metrics = run(profile("demo"), arguments(self.root))
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)
        self.assertEqual(observed, ["inference committed", "dataset finalized"])
        self.assertEqual((metrics["captured_pairs"], metrics["processed_pairs"]), (1, 1))
        self.assertEqual(json.loads((self.root / "meta/ego2.json").read_text())["recording_status"], "complete")
        frames = list(DatasetReplay(self.root).frames())
        self.assertEqual([frame[0].sequence for frame in frames], [1])
        self.assertTrue(frames[0][3]["observation.inference_valid"])
        source.close.assert_called_once()
        pipeline.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
