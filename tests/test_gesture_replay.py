"""Recorded replay checks use tiny local files and never start a viewer."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ego2.common.sources import Pair
from ego2.gesture.replay import DatasetReplay, VIDEO_KEYS, replay
from ego2.gesture.tracking import Hand


def fixture(root, *, episodes=2):
    """Two episodes share files: exercise the v3 offsets absent from v2."""
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    profile = dict(channels=[dict(camera="cam0"), dict(camera="cam1")],
                   viewer=dict(memory_limit="128MB"))
    features = {key: dict(dtype="video", shape=[24, 32, 3],
                         info={"video.fps": 10}) for key in VIDEO_KEYS}
    info = dict(codebase_version="v3.0", fps=10, total_frames=episodes * 2,
                total_episodes=episodes, features=features,
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/ego2.json").write_text(json.dumps(dict(
        schema_version=1, profile=profile, recording_status="complete")))
    for eye, key in enumerate(VIDEO_KEYS):
        path = root / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
        path.parent.mkdir(parents=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
        if not writer.isOpened():
            raise RuntimeError("OpenCV mp4v encoder unavailable for replay fixture")
        for index in range(episodes * 2):
            writer.write(np.full((24, 32, 3), 20 + index * 40 + eye * 10, np.uint8))
        writer.release()
    metadata, rows = [], []
    hand = dict(id=4, points=[[10., 12.]] * 20 + [[None, None]],
                visible=[True] * 20 + [False], handedness="Left", gesture="Open_Palm",
                score=0.875, pinch="Pinching", pinch_ratio=0.15)
    for episode in range(episodes):
        entry = dict(episode_index=episode, length=2, dataset_from_index=episode * 2,
                     dataset_to_index=episode * 2 + 2,
                     **{"data/chunk_index": 0, "data/file_index": 0})
        for key in VIDEO_KEYS:
            entry.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                          f"videos/{key}/from_timestamp": episode * 0.2,
                          f"videos/{key}/to_timestamp": (episode + 1) * 0.2})
        metadata.append(entry)
        for frame in range(2):
            index = episode * 2 + frame
            rows.append(dict(
                index=index, episode_index=episode, frame_index=frame, timestamp=frame / 10,
                task_index=0,
                **{"observation.source_timestamp": 50 + index * 0.17,
                   "observation.receive_timestamp": 100 + index * 0.18,
                   "observation.capture_time": index * 0.17,
                   "observation.source_sequence": index + 10,
                   "observation.inference_valid": frame == 0,
                   "observation.inference_timestamp": index * 0.17 if frame == 0 else None,
                   "observation.processing_ms": 3.25 if frame == 0 else None,
                   "observation.gesture_json": json.dumps(dict(
                       hands=[[hand], []] if frame == 0 else [[], []], detections=None)),
                   "observation.capture_metadata_json": "{}"}))
    pq.write_table(pa.Table.from_pylist(metadata), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    return info, rows


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "dataset"
        self.info, self.rows = fixture(self.root)

    def rewrite_rows(self, rows):
        pq.write_table(pa.Table.from_pylist(rows), self.root / "data/chunk-000/file-000.parquet")

    def args(self, **overrides):
        args = dict(replay=self.root, episode=None, frames=0, seconds=0,
                    realtime=False, headless=True, save=Path(self.tmp.name) / "replay.rrd",
                    connect=None, web=False, web_port=9090, grpc_port=9876, metrics=None)
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_shared_file_offsets_restore_original_time_and_hand_values(self):
        frames = list(DatasetReplay(self.root).frames())
        self.assertEqual(len(frames), 4)
        for index, (pair, stamp, hands, row) in enumerate(frames):
            self.assertEqual(pair.sequence, index + 10)
            self.assertAlmostEqual(pair.stamp, 50 + index * 0.17)
            self.assertAlmostEqual(pair.received, 100 + index * 0.18)
            self.assertAlmostEqual(stamp, index * 0.17)
            for eye, image in enumerate(pair.images):
                self.assertLess(abs(float(image.mean()) - (20 + index * 40 + eye * 10)), 5)
            if index % 2:
                self.assertEqual(hands, [[], []])
                self.assertFalse(row["observation.inference_valid"])
            else:
                hand = hands[0][0]
                self.assertEqual((hand.id, hand.gesture, hand.pinch), (4, "Open_Palm", "Pinching"))
                self.assertEqual(hand.score, 0.875)
                self.assertTrue(np.isnan(hand.points[-1]).all())
                self.assertFalse(hand.visible[-1])

    def test_episode_selection_uses_video_offset_and_keeps_capture_timeline(self):
        frames = list(DatasetReplay(self.root).frames(1))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][0].sequence, 12)
        self.assertAlmostEqual(frames[0][1], 0.34)
        self.assertLess(abs(float(frames[0][0].images[0].mean()) - 100), 5)
        with self.assertRaisesRegex(ValueError, "Episode 9 does not exist"):
            list(DatasetReplay(self.root).frames(9))

    def test_missing_saved_profile_has_clear_error(self):
        (self.root / "meta/ego2.json").write_text('{"schema_version":1}')
        with self.assertRaisesRegex(ValueError, "Missing saved profile"):
            DatasetReplay(self.root)

    def test_missing_video_has_clear_error(self):
        path = next((self.root / "videos").rglob("*.mp4"))
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing dataset video"):
            list(DatasetReplay(self.root).frames())

    def test_truncated_video_count_is_rejected(self):
        path = next((self.root / "videos").rglob("*.mp4"))
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
        writer.write(np.zeros((24, 32, 3), np.uint8))
        writer.release()
        with self.assertRaisesRegex(ValueError, "Video frame count.*shorter"):
            list(DatasetReplay(self.root).frames())

    def test_missing_data_row_is_rejected(self):
        self.rewrite_rows(self.rows[:1] + self.rows[2:])
        with self.assertRaisesRegex(ValueError, "has 1 data frames; expected 2"):
            list(DatasetReplay(self.root).frames())

    def test_corrupt_gesture_and_backwards_time_are_rejected(self):
        original = self.rows[0]["observation.gesture_json"]
        self.rows[0]["observation.gesture_json"] = "not JSON"
        self.rewrite_rows(self.rows)
        with self.assertRaisesRegex(ValueError, "Invalid observation.gesture_json"):
            list(DatasetReplay(self.root).frames())
        self.rows[0]["observation.gesture_json"] = original
        self.rows[2]["observation.capture_time"] = 0
        self.rewrite_rows(self.rows)
        with self.assertRaisesRegex(ValueError, "goes backwards"):
            list(DatasetReplay(self.root).frames())

    def test_headless_replay_writes_valid_standalone_rrd(self):
        from ego2.gesture.visualizer import Visualizer
        args = self.args()
        with patch.object(Visualizer, "_start_web", side_effect=AssertionError("No browser allowed")), \
                patch("rerun.RecordingStream.spawn", side_effect=AssertionError("No viewer allowed")):
            self.assertEqual(replay(args), dict(replayed_pairs=4, inference_pairs=2,
                                                missing_inference_pairs=2))
        self.assertGreater(args.save.stat().st_size, 500)
        if shutil.which("rerun"):
            checked = subprocess.run(["rerun", "rrd", "verify", str(args.save)],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stderr + checked.stdout)

    def test_frame_limit_does_not_decode_a_fifth_or_extra_frame(self):
        from ego2.gesture.visualizer import Visualizer
        args = self.args(frames=1)
        with patch.object(Visualizer, "_start_web", side_effect=AssertionError("No browser allowed")):
            self.assertEqual(replay(args)["replayed_pairs"], 1)

    def test_cli_replay_never_loads_current_config_model_or_camera(self):
        from ego2.gesture.__main__ import main
        output = Path(self.tmp.name) / "cli.rrd"
        with patch("ego2.gesture.__main__.load_config", side_effect=AssertionError("No current config")), \
                patch("ego2.common.sources.Source.open", side_effect=AssertionError("No camera")), \
                patch("ego2.gesture.recognizer.Recognizer.__init__", side_effect=AssertionError("No model")):
            self.assertEqual(main(["--replay", str(self.root), "--episode", "1", "--frames", "1",
                                   "--headless", "--save", str(output),
                                   "--config", "/nonexistent/config.yaml", "--model", "/nonexistent/model.task"]), 0)
        self.assertGreater(output.stat().st_size, 500)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for dataset recording")
    def test_real_writer_roundtrip_is_lossless_and_preserves_sparse_inference(self):
        from ego2.gesture.dataset import GestureDataset
        root = Path(self.tmp.name) / "writer-dataset"
        profile = dict(channels=[dict(camera="cam0"), dict(camera="cam1")],
                       input=dict(type="demo"))
        writer = GestureDataset(root, profile, {}, fps=10, episode_frames=2)
        rng = np.random.default_rng(123)
        pairs = [Pair(tuple(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(2)),
                      123. + index * 0.15, 400. + index * 0.16, index + 1,
                      metadata={"sensor_frame": index}, raw_transport=np.arange(32, dtype=np.uint8))
                 for index in range(3)]
        hand = Hand(7, np.full((21, 2), [10., 12.]), np.ones(21, bool),
                    "Left", "Open_Palm", 0.92, "Pinching", 0.11)
        try:
            for index, pair in enumerate(pairs):
                writer.capture(pair)
                if index != 1:
                    writer.result(pair, index * 0.15, [[hand], []],
                                  detections=[[{"handedness_score": 0.99}], []], processing_ms=2.5)
        finally:
            writer.close()
        frames = list(DatasetReplay(root).frames())
        self.assertEqual(len(frames), 3)
        for index, (pair, stamp, hands, row) in enumerate(frames):
            for actual, expected in zip(pair.images, pairs[index].images):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(pair.metadata["sensor_frame"], index)
            self.assertAlmostEqual(stamp, index * 0.15)
            self.assertEqual(row["observation.inference_valid"], index != 1)
            self.assertEqual(row["observation.raw_transport_length"], 32)
            if index != 1:
                self.assertEqual(hands[0][0].id, 7)
                details = json.loads(row["observation.gesture_json"])
                self.assertEqual(details["detections"][0][0]["handedness_score"], 0.99)


if __name__ == "__main__":
    unittest.main()
