"""Recording contract checks using actual FFmpeg and Arrow readers."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ego2.common.sources import Pair
from ego2.gesture.dataset import GestureDataset, VIDEO_KEYS
from ego2.gesture.tracking import Hand


AVAILABLE = all(importlib.util.find_spec(module) for module in ("pyarrow", "pandas")) and bool(shutil.which("ffmpeg"))


def make_pair(sequence):
    image = np.random.default_rng(sequence).integers(0, 256, (18, 24, 3), dtype=np.uint8)
    return Pair((image, image[:, ::-1].copy()), 400 + sequence * 0.037, 900 + sequence * 0.04,
                sequence, metadata={"test": sequence, "imu_samples": [{"timestamp_seconds": sequence / 100}]},
                raw_transport=np.arange(48, dtype=np.uint8).reshape(6, 4, 2))


def make_hand():
    points = np.arange(42, dtype=np.float32).reshape(21, 2)
    points[3] = np.nan
    visible = np.ones(21, dtype=bool)
    visible[3] = False
    return Hand(7, points, visible, "Left", "Open_Palm", 0.9, "Open", None)


@unittest.skipUnless(AVAILABLE, "recording needs pyarrow, pandas and ffmpeg")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        import pyarrow.parquet as pq
        self.pq = pq
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        self.profile = {"input": {"type": "demo"}, "channels": [{"camera": "cam0"}, {"camera": "cam1"}]}

    def writer(self, **kwargs):
        return GestureDataset(self.root, self.profile, {}, 30, task="test gesture", **kwargs)

    def frame_rows(self):
        return [row for path in sorted((self.root / "data").glob("*/*.parquet"))
                for row in self.pq.read_table(path).to_pylist()]

    def test_real_lossless_video_sparse_annotations_and_v3_shards(self):
        writer = self.writer(episode_frames=2)
        pairs = [make_pair(i) for i in (1, 2, 4, 5, 7)]
        for pair in pairs:
            writer.capture(pair)
        # Simulate slower inference arriving after a segment's encoder was closed.
        writer.result(pairs[0], 0.0, [[make_hand()], []],
                      detections=[[{"world_points": [[0.1, 0.2, 0.3]], "score": float("nan")}], []],
                      processing_ms=12)
        writer.result(pairs[-1], 0.222, [[], []], processing_ms=5)
        writer.close()
        writer.close()
        info = json.loads((self.root / "meta/info.json").read_text())
        self.assertEqual(info["codebase_version"], "v3.0")
        self.assertEqual((info["total_episodes"], info["total_frames"]), (3, 5))
        self.assertEqual(info["splits"], {"train": "0:3"})
        rows = self.frame_rows()
        self.assertEqual([r["observation.source_sequence"] for r in rows], [1, 2, 4, 5, 7])
        self.assertEqual([r["observation.inference_valid"] for r in rows], [True, False, False, False, True])
        self.assertEqual([r["frame_index"] for r in rows], [0, 1, 0, 1, 0])
        self.assertEqual([r["index"] for r in rows], list(range(5)))
        np.testing.assert_allclose([r["timestamp"] for r in rows], [0, 1/30, 0, 1/30, 0])
        np.testing.assert_allclose([r["observation.capture_time"] for r in rows], [p.stamp - pairs[0].stamp for p in pairs])
        self.assertNotIn(VIDEO_KEYS[0], rows[0])
        self.assertEqual(rows[0]["observation.hands.eye0.id"], [7, 0])
        self.assertEqual(rows[0]["observation.hands.eye0.present"], [True, False])
        self.assertEqual(rows[0]["observation.hands.eye0.pinch_ratio_valid"], [False, False])
        self.assertFalse(rows[0]["observation.hands.eye0.visible"][3])
        self.assertEqual(json.loads(rows[0]["observation.gesture_json"])["hands"][0][0]["points"][3], [None, None])
        self.assertEqual(json.loads(rows[0]["observation.gesture_json"])["detections"][0][0]["score"], None)
        episodes = [row for path in sorted((self.root / "meta/episodes").glob("*/*.parquet"))
                    for row in self.pq.read_table(path).to_pylist()]
        self.assertEqual([e["dataset_from_index"] for e in episodes], [0, 2, 4])
        self.assertEqual([e["dataset_to_index"] for e in episodes], [2, 4, 5])
        for episode in episodes:
            start, end = episode["dataset_from_index"], episode["dataset_to_index"]
            for eye, key in enumerate(VIDEO_KEYS):
                path = self.root / info["video_path"].format(video_key=key,
                    chunk_index=episode[f"videos/{key}/chunk_index"], file_index=episode[f"videos/{key}/file_index"])
                decoder = cv2.VideoCapture(str(path))
                try:
                    self.assertTrue(decoder.isOpened())
                    for pair in pairs[start:end]:
                        ok, image = decoder.read()
                        self.assertTrue(ok)
                        np.testing.assert_array_equal(image, pair.images[eye])
                    self.assertFalse(decoder.read()[0])
                finally:
                    decoder.release()
        import pandas as pd
        tasks = pd.read_parquet(self.root / "meta/tasks.parquet")
        self.assertEqual(tasks.iloc[0].name, "test gesture")
        self.assertEqual(tasks.iloc[0].task_index, 0)
        schema = self.pq.read_schema(self.root / "data/chunk-000/file-000.parquet")
        self.assertEqual(str(schema.field("timestamp").type), "float")
        self.assertEqual(str(schema.field("episode_index").type), "int64")
        self.assertEqual(schema.field("observation.hands.eye0.points").type.list_size, 84)
        stats = json.loads((self.root / "meta/stats.json").read_text())
        self.assertEqual(np.array(stats[VIDEO_KEYS[0]]["mean"]).shape, (3, 1, 1))
        self.assertEqual(stats[VIDEO_KEYS[0]]["count"], [5])
        self.assertEqual(stats["index"]["count"], [5])
        self.assertAlmostEqual(stats["index"]["mean"][0], 2)
        for values in stats.values():
            for value in values.values():
                self.assertTrue(np.isfinite(np.asarray(value)).all())
        self.assertEqual(json.loads((self.root / "meta/ego2.json").read_text())["recording_status"], "complete")

    def test_raw_transport_byte_ranges_and_metadata_survive(self):
        writer = self.writer(episode_frames=2)
        pairs = [make_pair(i) for i in (1, 2, 3)]
        for pair in pairs:
            writer.capture(pair)
        writer.close()
        for row, pair in zip(self.frame_rows(), pairs):
            metadata = json.loads(row["observation.capture_metadata_json"])
            self.assertEqual(metadata["imu_samples"], pair.metadata["imu_samples"])
            descriptor = metadata["raw_transport"]
            with (self.root / row["observation.raw_transport_path"]).open("rb") as handle:
                handle.seek(row["observation.raw_transport_offset"])
                payload = handle.read(row["observation.raw_transport_length"])
            restored = np.frombuffer(payload, dtype=descriptor["dtype"]).reshape(descriptor["shape"])
            np.testing.assert_array_equal(restored, pair.raw_transport)
        self.assertEqual([r["observation.raw_transport_offset"] for r in self.frame_rows()], [0, 48, 0])

    def test_failure_finalizes_available_capture_and_preserves_reason(self):
        writer = self.writer()
        writer.capture(make_pair(1))
        writer.close(error=RuntimeError("model stopped"))
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "failed")
        self.assertEqual(extra["error"], "model stopped")
        self.assertEqual(len(self.frame_rows()), 1)
        self.assertFalse(self.frame_rows()[0]["observation.inference_valid"])

    def test_encoder_close_failure_publishes_only_successful_prefix(self):
        from ego2.gesture.replay import DatasetReplay
        writer = self.writer(episode_frames=2)
        pairs = [make_pair(i) for i in (1, 2, 3)]
        for pair in pairs:
            writer.capture(pair)
            writer.result(pair, pair.stamp - pairs[0].stamp, [[], []])
        video = writer._videos[0]
        real_close = video.close
        original_error = RuntimeError("last segment encoder failed")
        def fail_close():
            real_close()
            raise original_error
        with patch.object(video, "close", side_effect=fail_close):
            with self.assertRaises(RuntimeError) as raised:
                writer.close()
        self.assertIs(raised.exception, original_error)
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "failed")
        self.assertEqual(extra["error"], str(original_error))
        self.assertEqual((extra["capture_count"], extra["inference_count"], extra["finalized_frames"]), (3, 3, 2))
        info = json.loads((self.root / "meta/info.json").read_text())
        self.assertEqual((info["total_frames"], info["total_episodes"]), (2, 1))
        for journal in ("capture", "inference"):
            self.assertEqual(len((self.root / f"raw/{journal}.jsonl").read_text().splitlines()), 3)
        frames = list(DatasetReplay(self.root).frames())
        self.assertEqual([item[0].sequence for item in frames], [1, 2])
        for item, pair in zip(frames, pairs):
            np.testing.assert_array_equal(item[0].images[0], pair.images[0])
        self.assertFalse((self.root / "data/chunk-000/file-001.parquet").exists())

    def test_low_disk_stops_before_writing_and_keeps_committed_prefix(self):
        writer = self.writer(min_free_bytes=1000)
        writer.capture(make_pair(1))
        writer.result(make_pair(1), 0, [[], []])
        # Inference may get ahead of the asynchronous storage worker.
        writer.result(make_pair(2), 1 / 30, [[], []])
        from types import SimpleNamespace
        with patch("ego2.gesture.dataset.shutil.disk_usage", return_value=SimpleNamespace(free=999)):
            with self.assertRaisesRegex(RuntimeError, "Low disk space"):
                writer.capture(make_pair(2))
        writer.close()
        self.assertEqual([row["observation.source_sequence"] for row in self.frame_rows()], [1])
        self.assertEqual(len((self.root / "raw/capture.jsonl").read_text().splitlines()), 1)
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "failed")
        self.assertIn("Low disk space", extra["error"])

    def test_successful_close_reports_sensor_gaps_and_missing_inference(self):
        writer = self.writer()
        first = replace(make_pair(1), metadata={"source_fps": 30,
            "image_timestamp_seconds": 10.0, "imu_samples": [{"timestamp_seconds": 9.99}]})
        second = replace(make_pair(2), metadata={"source_fps": 30,
            "image_timestamp_seconds": 10.6334, "imu_samples": [{"timestamp_seconds": 10.595}]})
        writer.capture(first)
        writer.capture(second)
        writer.result(first, 0, [[], []])
        writer.close()
        quality = json.loads((self.root / "meta/quality.json").read_text())
        self.assertEqual(quality["sensor_continuity"], "issues_detected")
        self.assertEqual(quality["device_rgb"]["estimated_missing_samples"], 18)
        self.assertEqual(quality["imu"]["estimated_missing_samples"], 120)
        self.assertEqual(quality["inference"], dict(captured_frames=2, inferred_frames=1, uninferred_frames=1))
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "complete")
        self.assertEqual(extra["quality"], quality)

    def test_partial_stereo_write_never_publishes_misaligned_segment(self):
        writer = self.writer(episode_frames=2)
        for sequence in (1, 2, 3):
            writer.capture(make_pair(sequence))
        # The left encoder accepts frame 4, but the right writer fails.
        with patch.object(writer._videos[1], "write", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(OSError, "write failed"):
                writer.capture(make_pair(4))
        with self.assertRaisesRegex(OSError, "write failed"):
            writer.close()
        self.assertEqual([row["observation.source_sequence"] for row in self.frame_rows()], [1, 2])
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual((extra["capture_count"], extra["finalized_frames"]), (3, 2))
        self.assertEqual(extra["recording_status"], "failed")

    def test_encoder_failure_during_rollover_keeps_previous_segments_readable(self):
        writer = self.writer(episode_frames=1)
        writer.capture(make_pair(1))
        writer.capture(make_pair(2))
        video = writer._videos[0]
        real_close = video.close
        def fail_close():
            real_close()
            raise RuntimeError("rollover encoder failed")
        with patch.object(video, "close", side_effect=fail_close):
            with self.assertRaisesRegex(RuntimeError, "rollover encoder failed"):
                writer.capture(make_pair(3))
        with self.assertRaisesRegex(RuntimeError, "rollover encoder failed"):
            writer.close()
        self.assertEqual([row["observation.source_sequence"] for row in self.frame_rows()], [1])
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual((extra["capture_count"], extra["finalized_frames"]), (2, 1))
        self.assertEqual(extra["recording_status"], "failed")

    def test_overwrite_rejected_and_shape_change_not_written(self):
        writer = self.writer()
        pair = make_pair(1)
        writer.capture(pair)
        with self.assertRaises(FileExistsError):
            self.writer()
        with self.assertRaisesRegex(ValueError, "resolution changed"):
            writer.capture(replace(make_pair(2), images=(pair.images[0][:-1], pair.images[1])))
        writer.close()
        self.assertEqual(len(self.frame_rows()), 1)
        self.assertEqual(json.loads((self.root / "meta/ego2.json").read_text())["recording_status"], "failed")

    def test_repeated_or_backwards_capture_time_rejected_before_writing(self):
        from ego2.gesture.replay import DatasetReplay
        writer = self.writer()
        pair = make_pair(1)
        writer.capture(pair)
        for stamp in (pair.stamp, pair.stamp - 0.01):
            with self.subTest(stamp=stamp):
                with self.assertRaisesRegex(ValueError, "timestamps must be strictly increasing"):
                    writer.capture(replace(make_pair(2), stamp=stamp))
        writer.close()
        self.assertEqual([row["observation.source_sequence"] for row in self.frame_rows()], [1])
        self.assertEqual(len(list(DatasetReplay(self.root).frames())), 1)
        self.assertEqual(len((self.root / "raw/capture.jsonl").read_text().splitlines()), 1)
        raw_path = self.root / self.frame_rows()[0]["observation.raw_transport_path"]
        self.assertEqual(raw_path.stat().st_size, pair.raw_transport.nbytes)
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "failed")
        self.assertEqual((extra["capture_count"], extra["finalized_frames"]), (1, 1))

    def test_mismatched_annotation_rejected_during_finalization(self):
        writer = self.writer()
        writer.capture(make_pair(2))
        writer.result(make_pair(1), 0, [[], []])
        with self.assertRaisesRegex(ValueError, "was not captured"):
            writer.close()
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "failed")

    def test_failed_inference_partial_journal_tail_preserves_committed_annotations(self):
        writer = self.writer()
        for sequence in (1, 2):
            writer.capture(make_pair(sequence))
        writer.result(make_pair(1), 0, [[], []])
        original_write = writer._results.write
        partial = '{"sequence":'
        def fail_write(value):
            original_write(partial)
            writer._results.flush()
            raise OSError("partial inference journal write")
        with patch.object(writer._results, "write", side_effect=fail_write):
            with self.assertRaisesRegex(OSError, "partial inference journal write"):
                writer.result(make_pair(2), .037, [[], []])
        writer.close()
        rows = self.frame_rows()
        self.assertEqual([r["observation.source_sequence"] for r in rows], [1, 2])
        self.assertEqual([r["observation.inference_valid"] for r in rows], [True, False])
        self.assertTrue((self.root / "raw/inference.jsonl").read_text().endswith(partial))
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "failed")
        self.assertIn("partial inference journal write", extra["error"])
        self.assertEqual((extra["inference_count"], extra["finalized_inference_frames"]), (1, 1))
        quality = json.loads((self.root / "meta/quality.json").read_text())
        self.assertEqual(quality["inference"], dict(captured_frames=2, inferred_frames=1, uninferred_frames=1))

    def test_failed_segment_partial_journal_tail_publishes_only_committed_prefix(self):
        from ego2.gesture.replay import DatasetReplay
        writer = self.writer(episode_frames=2)
        for sequence in (1, 2, 3, 4):
            writer.capture(make_pair(sequence))
            writer.result(make_pair(sequence), (sequence - 1) * .037, [[], []])
        self.assertEqual(writer.episode_count, 1)
        original_write = writer._segments.write
        partial = '{"episode_index":'
        def fail_write(value):
            original_write(partial)
            writer._segments.flush()
            raise OSError("partial segment journal write")
        with patch.object(writer._segments, "write", side_effect=fail_write):
            with self.assertRaisesRegex(OSError, "partial segment journal write"):
                writer.close()
        self.assertEqual([r["observation.source_sequence"] for r in self.frame_rows()], [1, 2])
        self.assertTrue((self.root / "raw/segments.jsonl").read_text().endswith(partial))
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "failed")
        self.assertEqual((extra["capture_count"], extra["finalized_frames"]), (4, 2))
        self.assertEqual((extra["inference_count"], extra["finalized_inference_frames"]), (4, 2))
        quality = json.loads((self.root / "meta/quality.json").read_text())
        self.assertEqual(quality["capture_rows"], 2)
        self.assertEqual(quality["inference"], dict(captured_frames=2, inferred_frames=2, uninferred_frames=0))
        self.assertEqual([pair.sequence for pair, _, _, _ in DatasetReplay(self.root).frames()], [1, 2])


if __name__ == "__main__":
    unittest.main()
