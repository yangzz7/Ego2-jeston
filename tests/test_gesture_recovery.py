"""Recovery preserves source evidence and reconstructs only recorded samples."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from ego2.common.cyper_metadata import CyperMetadataDecoder
from ego2.common.sources import Pair, decode_cyper, white_balance
from ego2.gesture.dataset import GestureDataset, VIDEO_KEYS, VIDEO_PATH, _Video
from ego2.gesture.recovery import recover_dataset
from ego2.gesture.replay import DatasetReplay
from ego2.gesture.tracking import Hand


AVAILABLE = all(importlib.util.find_spec(name) for name in ("pyarrow", "pandas")) and bool(shutil.which("ffmpeg"))


@unittest.skipUnless(AVAILABLE, "recovery requires pyarrow, pandas, and ffmpeg")
class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.destination = Path(self.temp.name) / "recovered"
        self.profile = dict(input=dict(type="cyper", width=160, height=18, fps=30,
                                       bayer="auto", white_balance=True),
                            channels=[dict(camera="cam0"), dict(camera="cam1")])

    def pairs(self):
        result = []
        decoder = CyperMetadataDecoder()
        for index in range(5):
            raw = np.random.default_rng(index).integers(0, 256, (18, 160, 2), dtype=np.uint8)
            words = np.zeros(160, dtype=np.uint16)
            words[:5] = [2, 5, 0, 1000 + index * 333, 10]
            words[5:14] = [1000 + index * 333, 10, 100, 200, 300, 400, 500, 600, 16]
            raw[-1, :, 0], raw[-1, :, 1] = words & 255, words >> 8
            images, _ = decode_cyper(raw, 160, 18)
            images = tuple(white_balance(image) for image in images)
            result.append(Pair(images, 100 + index / 30, 200 + index / 30, index + 1,
                               decoder.decode(raw, 160, 18), raw))
        return result

    def record(self, raw=True):
        pairs = self.pairs()
        writer = GestureDataset(self.source, self.profile, {}, 30, task="original task", episode_frames=3)
        for pair in pairs:
            writer.capture(pair if raw else replace(pair, raw_transport=None))
        points = np.arange(42, dtype=np.float32).reshape(21, 2)
        hand = Hand(7, points, np.ones(21, dtype=bool), "Left", "Open_Palm", 0.9, "Open", 0.7)
        writer.result(pairs[0], 0, [[hand], []], detections=[[dict(label="first")], []], processing_ms=4)
        writer.result(pairs[-1], 4 / 30, [[], []], processing_ms=5)
        writer.close()
        return pairs

    def video(self, eye=1, episode=1):
        chunk, file = divmod(episode, 1000)
        return self.source / VIDEO_PATH.format(video_key=VIDEO_KEYS[eye], chunk_index=chunk, file_index=file)

    def damage(self):
        self.video().write_bytes(b"unfinished mp4 without moov")
        segments = self.source / "raw/segments.jsonl"
        segments.write_text(segments.read_text().splitlines()[0] + "\n")
        metadata_path = self.source / "meta/ego2.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(recording_status="failed", error="original viewer connection failed", finalized_frames=0)
        # Recovery must not inspect the current config/model paths.
        metadata["profile"]["_path"] = "/nonexistent/original/profile.yaml"
        metadata["profile"]["gesture"] = dict(model="/nonexistent/original/model.task")
        metadata_path.write_text(json.dumps(metadata))

    def fingerprints(self):
        return {str(path.relative_to(self.source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.source.rglob("*") if path.is_file()}

    def test_recovers_bad_last_video_exact_rgb_imu_and_sparse_inference(self):
        pairs = self.record()
        self.damage()
        snapshot = self.source / "meta/calibration/original.yaml"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text("saved_original: true\n")
        metadata_path = self.source / "meta/ego2.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["calibration_files"] = {"/no/longer/available.yaml": "meta/calibration/original.yaml"}
        metadata_path.write_text(json.dumps(metadata))
        before = self.fingerprints()
        report = recover_dataset(self.source, self.destination)
        self.assertEqual(before, self.fingerprints())
        self.assertEqual((report["capture_count"], report["inference_count"], report["missing_inference_count"]), (5, 2, 3))
        self.assertEqual([item["action"] for item in report["videos"]],
                         ["validated_and_hardlinked"] * 3 + ["regenerated_from_raw"])
        frames = list(DatasetReplay(self.destination).frames())
        self.assertEqual(len(frames), 5)
        for (pair, _, _, row), original in zip(frames, pairs):
            for image, expected in zip(pair.images, original.images):
                np.testing.assert_array_equal(image, expected)
            self.assertEqual(pair.sequence, original.sequence)
            self.assertEqual(pair.metadata["imu_samples"], original.metadata["imu_samples"])
        self.assertEqual([frame[3]["observation.inference_valid"] for frame in frames], [True, False, False, False, True])
        self.assertEqual(frames[0][2][0][0].id, 7)
        self.assertEqual(frames[1][2], [[], []])
        for name in ("capture.jsonl", "inference.jsonl"):
            self.assertEqual((self.source / "raw" / name).read_bytes(), (self.destination / "raw" / name).read_bytes())
        for relative in ("raw/transport/chunk-000/file-001.bin",
                         str(self.video(eye=0).relative_to(self.source))):
            self.assertEqual((self.source / relative).stat().st_ino, (self.destination / relative).stat().st_ino)
        relative = self.video().relative_to(self.source)
        self.assertNotEqual((self.source / relative).stat().st_ino, (self.destination / relative).stat().st_ino)
        extra = json.loads((self.destination / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "complete")
        self.assertNotIn("error", extra)
        self.assertEqual(extra["recovery"]["source_error"], "original viewer connection failed")
        self.assertEqual(extra["finalized_frames"], 5)
        self.assertEqual(snapshot.read_bytes(), (self.destination / "meta/calibration/original.yaml").read_bytes())
        import pandas as pd
        self.assertEqual(pd.read_parquet(self.destination / "meta/tasks.parquet").index.tolist(), ["original task"])

    def test_corrupt_video_without_raw_fails_and_preserves_source(self):
        self.record(raw=False)
        self.damage()
        before = self.fingerprints()
        with self.assertRaisesRegex(ValueError, "raw transport is missing.*--no-raw"):
            recover_dataset(self.source, self.destination)
        self.assertEqual(before, self.fingerprints())
        extra = json.loads((self.destination / "meta/ego2.json").read_text())
        self.assertEqual(extra["recording_status"], "recovery_failed")
        self.assertFalse((self.destination / "meta/info.json").exists())

    def test_decodable_short_video_is_rebuilt_after_full_frame_validation(self):
        pairs = self.record()
        path = self.source / "short.mp4"
        encoder = _Video(path, pairs[-2].images[1].shape, 30, shutil.which("ffmpeg"))
        encoder.write(pairs[-2].images[1])
        encoder.close()
        path.replace(self.video())
        report = recover_dataset(self.source, self.destination)
        self.assertEqual(report["videos"][-1]["action"], "regenerated_from_raw")
        self.assertIn("frame 1", report["videos"][-1]["original_error"])
        frames = list(DatasetReplay(self.destination).frames())
        np.testing.assert_array_equal(frames[-1][0].images[1], pairs[-1].images[1])

    def test_valid_rgb_without_raw_is_recovered_without_fabricating_transport(self):
        self.record(raw=False)
        report = recover_dataset(self.source, self.destination)
        self.assertTrue(all(video["action"] == "validated_and_hardlinked" for video in report["videos"]))
        frames = list(DatasetReplay(self.destination).frames())
        self.assertTrue(all(frame[3]["observation.raw_transport_length"] == 0 for frame in frames))

    def test_rejects_raw_range_beyond_file_before_creating_destination(self):
        self.record()
        path = self.source / "raw/transport/chunk-000/file-001.bin"
        with path.open("r+b") as handle:
            handle.truncate(path.stat().st_size - 1)
        before = self.fingerprints()
        with self.assertRaisesRegex(ValueError, "truncated raw transport"):
            recover_dataset(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(before, self.fingerprints())

    def test_rejects_reordered_capture_and_unknown_inference(self):
        self.record()
        capture_path = self.source / "raw/capture.jsonl"
        original = capture_path.read_text()
        lines = original.splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        capture_path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "capture indices are not contiguous"):
            recover_dataset(self.source, self.destination)
        capture_path.write_text(original)
        result_path = self.source / "raw/inference.jsonl"
        rows = [json.loads(line) for line in result_path.read_text().splitlines()]
        rows[-1]["sequence"] = 999
        result_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        with self.assertRaisesRegex(ValueError, "was not captured"):
            recover_dataset(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_no_info_or_final_segments_can_be_reconstructed_from_journals(self):
        self.record()
        (self.source / "meta/info.json").unlink()
        (self.source / "meta/tasks.parquet").unlink()
        (self.source / "raw/segments.jsonl").unlink()
        report = recover_dataset(self.source, self.destination)
        self.assertEqual(report["capture_count"], 5)
        self.assertEqual(len(list(DatasetReplay(self.destination).frames())), 5)
        import pandas as pd
        self.assertEqual(pd.read_parquet(self.destination / "meta/tasks.parquet").index.tolist(), ["original task"])

    def test_existing_or_nested_destination_rejected(self):
        self.record()
        for destination in (self.source, self.source / "nested"):
            with self.assertRaises(FileExistsError):
                recover_dataset(self.source, destination)
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            recover_dataset(self.source, self.destination)


if __name__ == "__main__":
    unittest.main()
