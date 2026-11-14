"""Offline annotations are published together, with the live generation intact."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ego2.common.config import Camera
from ego2.common.sources import Pair
from ego2.gesture.dataset import GestureDataset
from ego2.gesture.reinfer import reinfer_dataset
from ego2.gesture.replay import DatasetReplay


AVAILABLE = all(importlib.util.find_spec(name) for name in ("pyarrow", "pandas")) and shutil.which("ffmpeg")


class FakePipeline:
    seen = []
    closed = False
    fail_at = None
    profile = None

    def __init__(self, profile, cameras):
        type(self).profile = profile
        assert isinstance(cameras["cam0"], Camera)
        assert isinstance(cameras["cam0"].K, np.ndarray)
        self.first = None
        self.observations = [[], []]

    def process(self, pair):
        self.seen.append(pair.sequence)
        if pair.sequence == self.fail_at:
            raise RuntimeError("inference failed")
        if self.first is None:
            self.first = pair.stamp
        self.observations = [[{"sequence": pair.sequence}], []]
        return pair.stamp - self.first, [[], []]

    def close(self):
        type(self).closed = True


@unittest.skipUnless(AVAILABLE, "reinference needs pyarrow, pandas and ffmpeg")
class ReinferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "recorded"
        self.model = self.base / "original.task"
        self.model.write_bytes(b"original saved model")
        profile = dict(input=dict(type="demo"), gesture=dict(model=str(self.model)),
            channels=[dict(camera="cam0"), dict(camera="cam1")])
        cameras = {name: Camera(name, np.eye(3), np.zeros(4), (32, 24), np.eye(4), 1.5)
                   for name in ("cam0", "cam1")}
        writer = GestureDataset(self.root, profile, cameras, fps=10, episode_frames=2,
                                task="recorded task")
        for index in range(4):
            image = np.full((24, 32, 3), index * 50, dtype=np.uint8)
            pair = Pair((image, image.copy()), 300 + index * .1, 500 + index * .1,
                        index + 1, metadata={"frame": index})
            writer.capture(pair)
            if index in (0, 3):
                writer.result(pair, index * .1, [[], []], detections=[[], []])
        writer.close()
        (self.root / "viewer.rrd").write_bytes(b"retained original recording")
        self.original_journal = (self.root / "raw/inference.jsonl").read_bytes()
        self.original_metadata = (self.root / "meta/ego2.json").read_bytes()
        self.original_meta_files = {path.relative_to(self.root / "meta"): path.read_bytes()
                                    for path in (self.root / "meta").rglob("*") if path.is_file()}
        FakePipeline.seen, FakePipeline.closed, FakePipeline.fail_at = [], False, None

    def snapshot(self):
        return {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.root.rglob("*") if path.is_file()}

    def run_reinfer(self):
        with patch("ego2.gesture.reinfer.Pipeline", FakePipeline), \
                patch("ego2.common.sources.Source.open", side_effect=AssertionError("No camera")):
            return reinfer_dataset(self.root)

    def test_all_frames_recomputed_with_live_journal_and_media_preserved(self):
        media = {p.relative_to(self.root): p.stat().st_ino
                 for p in self.root.rglob("*.mp4")}
        result = self.run_reinfer()
        self.assertEqual(FakePipeline.seen, [1, 2, 3, 4])
        self.assertTrue(FakePipeline.closed)
        self.assertEqual((result["previous_inference_count"], result["inference_count"],
                          result["filled_missing_frames"]), (2, 4, 2))
        self.assertEqual(result["model_sha256"], hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.assertEqual((self.root / "raw/inference_live.jsonl").read_bytes(), self.original_journal)
        self.assertEqual((self.root / "meta/ego2_live.json").read_bytes(), self.original_metadata)
        for relative, content in self.original_meta_files.items():
            self.assertEqual((self.root / "meta/live" / relative).read_bytes(), content)
        for relative, inode in media.items():
            self.assertEqual((self.root / relative).stat().st_ino, inode)
        frames = list(DatasetReplay(self.root).frames())
        self.assertEqual(len(frames), 4)
        for pair, stamp, _, row in frames:
            self.assertTrue(row["observation.inference_valid"])
            details = json.loads(row["observation.gesture_json"])
            self.assertEqual(details["detections"][0][0]["sequence"], pair.sequence)
            self.assertAlmostEqual(row["observation.inference_timestamp"], stamp)
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        self.assertEqual(extra["inference_count"], 4)
        self.assertEqual(extra["reinference"], result)
        self.assertEqual((self.root / "viewer.rrd").read_bytes(), b"retained original recording")
        with self.assertRaisesRegex(FileExistsError, "backup already exists"):
            self.run_reinfer()

    def test_pipeline_failure_leaves_every_original_file_unchanged(self):
        before = self.snapshot()
        FakePipeline.fail_at = 3
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_reinfer()
        self.assertEqual(FakePipeline.seen, [1, 2, 3])
        self.assertTrue(FakePipeline.closed)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(list(self.base.glob(".recorded.reinfer-*")), [])

    def test_atomic_commit_failure_leaves_original_tables_and_journal_intact(self):
        before = self.snapshot()
        with patch("ego2.gesture.reinfer._exchange_directories", side_effect=OSError("unsupported filesystem")):
            with self.assertRaisesRegex(OSError, "unsupported filesystem"):
                self.run_reinfer()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(list(self.base.glob(".recorded.reinfer-*")), [])

    def test_changed_model_is_rejected_before_inference(self):
        before = self.snapshot()
        self.model.write_bytes(b"new detector")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_reinfer()
        self.assertEqual(FakePipeline.seen, [])
        self.assertEqual(self.snapshot(), before)

    def test_saved_mask_preferred_over_changed_external_mask(self):
        extra = json.loads((self.root / "meta/ego2.json").read_text())
        external = self.base / "original-mask.png"
        external.write_bytes(b"changed outside dataset")
        saved = self.root / "meta/calibration/00-mask.png"
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"recording-time mask")
        extra["profile"]["channels"][0]["valid_mask"] = str(external)
        extra["calibration_files"][str(external)] = str(saved.relative_to(self.root))
        (self.root / "meta/ego2.json").write_text(json.dumps(extra))
        self.run_reinfer()
        self.assertEqual(FakePipeline.profile["channels"][0]["valid_mask"], str(saved))
        self.assertEqual(saved.read_bytes(), b"recording-time mask")


if __name__ == "__main__":
    unittest.main()
