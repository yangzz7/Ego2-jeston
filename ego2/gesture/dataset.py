"""Streaming, local LeRobot v3 capture without importing the training stack.

Capture runs before the latest-frame inference queue. Its durable JSONL journal
and lossless videos are merged with the (possibly sparse) inference journal on
close. Memory use does not grow with recording length. One bounded episode is
stored per v3 shard; offsets, rather than filenames, define episode boundaries.
"""
from dataclasses import asdict, is_dataclass
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .quality import CaptureQuality


DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_KEYS = ("observation.images.eye0", "observation.images.eye1")


def jsonable(value):
    """Keep every model field; represent unavailable/nonfinite values as null."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value):
    return json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _write_json(path, value):
    """An interrupted metadata update must not truncate the previous version."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


class _Moments:
    """Population moments, merging batches without accumulating frames."""
    def __init__(self):
        self.count = 0

    def add(self, array):
        a = np.asarray(array, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        if not len(a):
            return
        count, mean, variance = len(a), a.mean(axis=0), a.var(axis=0)
        low, high = a.min(axis=0), a.max(axis=0)
        if not self.count:
            self.count, self.mean, self.m2 = count, mean, variance * count
            self.low, self.high = low, high
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2 += variance * count + delta ** 2 * self.count * count / total
        self.mean += delta * count / total
        self.low = np.minimum(self.low, low)
        self.high = np.maximum(self.high, high)
        self.count = total

    def merge(self, stats):
        count = int(stats["count"][0])
        if not count:
            return
        mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
        low, high = np.asarray(stats["min"]), np.asarray(stats["max"])
        if not self.count:
            self.count, self.mean, self.m2 = count, mean.copy(), std ** 2 * count
            self.low, self.high = low.copy(), high.copy()
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2 += std ** 2 * count + delta ** 2 * self.count * count / total
        self.mean += delta * count / total
        self.low, self.high = np.minimum(self.low, low), np.maximum(self.high, high)
        self.count = total

    def stats(self, image=False, count=None):
        shape = (3, 1, 1) if image else self.mean.shape
        return dict(min=self.low.reshape(shape).tolist(), max=self.high.reshape(shape).tolist(),
                    mean=self.mean.reshape(shape).tolist(),
                    std=np.sqrt(np.maximum(self.m2 / self.count, 0)).reshape(shape).tolist(),
                    count=[self.count if count is None else count])


class _Video:
    def __init__(self, path, shape, fps, ffmpeg, *, isolate_signals=False):
        self.path, self.shape = path, tuple(shape)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.errors = tempfile.TemporaryFile(mode="w+b")
        h, w, _ = self.shape
        # RGB H.264 preserves all BGR input samples exactly (no YUV conversion).
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{w}x{h}",
                   "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264rgb",
                   "-crf", "0", "-preset", "ultrafast", "-threads", "1", "-g", "30",
                   "-bf", "0", str(path)]
        try:
            # Offline finalizers must finish after the application handles Ctrl+C.
            # Live capture encoders retain their worker's process-group lifecycle.
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=self.errors, start_new_session=isolate_signals)
        except BaseException:
            self.errors.close()
            raise
        self.closed = False

    def write(self, image):
        if image.dtype != np.uint8 or tuple(image.shape) != self.shape:
            raise ValueError("Capture images must remain uint8 BGR with a constant shape")
        try:
            self.process.stdin.write(np.ascontiguousarray(image).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"Video encoder failed: {self.path}") from exc

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError(f"Video encoder did not finish: {self.path}")
            if code:
                self.errors.seek(0)
                detail = self.errors.read(8192).decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Video encoder exited {code}: {detail}")
        finally:
            self.errors.close()


def _features():
    out = {}
    def add(key, dtype, size=1, names=None):
        out[key] = dict(dtype=dtype, shape=[size], names=names)
    add("timestamp", "float32")
    for key in ("frame_index", "episode_index", "index", "task_index"):
        add(key, "int64")
    for key in ("source_timestamp", "receive_timestamp", "capture_time", "inference_timestamp"):
        add("observation." + key, "float64")
    for key in ("source_sequence", "raw_transport_offset", "raw_transport_length"):
        add("observation." + key, "int64")
    add("observation.inference_valid", "bool")
    add("observation.processing_ms", "float32")
    for key in ("gesture_json", "capture_metadata_json", "raw_transport_path"):
        add("observation." + key, "string")
    for eye in range(2):
        prefix = f"observation.hands.eye{eye}."
        for name, dtype, length in (("present", "bool", 2), ("id", "int64", 2),
                                    ("points", "float32", 84), ("visible", "bool", 42),
                                    ("score", "float32", 2), ("score_valid", "bool", 2),
                                    ("pinch_ratio", "float32", 2), ("pinch_ratio_valid", "bool", 2)):
            add(prefix + name, dtype, length)
    return out


def _hand_values(hands):
    out = {}
    for eye in range(2):
        values = dict(present=np.zeros(2, bool), id=np.zeros(2, np.int64),
                      points=np.zeros((2, 21, 2), np.float32), visible=np.zeros((2, 21), bool),
                      score=np.zeros(2, np.float32), score_valid=np.zeros(2, bool),
                      pinch_ratio=np.zeros(2, np.float32), pinch_ratio_valid=np.zeros(2, bool))
        eye_hands = hands[eye]
        if len(eye_hands) > 2:
            raise ValueError("At most two hands per eye are supported")
        for i, hand in enumerate(eye_hands):
            points = np.asarray(hand["points"], dtype=np.float32)
            if points.shape != (21, 2):
                raise ValueError("Recorded Hand.points must contain 21 raw-image pixel XY coordinates")
            valid = np.isfinite(points).all(axis=1)
            values["present"][i] = True
            values["id"][i] = hand["id"]
            values["points"][i] = np.nan_to_num(points, nan=0, posinf=0, neginf=0)
            values["visible"][i] = np.asarray(hand["visible"], dtype=bool) & valid
            for key in ("score", "pinch_ratio"):
                value = hand.get(key)
                if value is not None and math.isfinite(float(value)):
                    values[key][i], values[key + "_valid"][i] = value, True
        out.update({f"observation.hands.eye{eye}.{name}": value.ravel().tolist()
                    for name, value in values.items()})
    return out


class GestureDataset:
    """Single capture-thread / single inference-thread recording.

    Stop and join the capture thread before ``close``. ``capture`` and ``result``
    each require strictly increasing source sequences. They write separate files
    and never wait for one another. Encoder backpressure preserves delivered
    source frames; frames lost inside a driver before Source.read are unknowable.
    """
    def __init__(self, root, profile, cameras, fps, task="gesture capture", episode_frames=900,
                 min_free_bytes=2 * 1024**3):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Dataset recording requires: pip install -r requirements-dataset.txt") from exc
        self.pa, self.pq, self.pd = pa, pq, pd
        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise RuntimeError("Dataset recording requires ffmpeg with the libx264rgb encoder in PATH")
        if not math.isfinite(float(fps)) or fps <= 0:
            raise ValueError("Dataset fps must be finite and positive")
        if isinstance(episode_frames, bool) or int(episode_frames) != episode_frames or episode_frames < 1:
            raise ValueError("episode_frames must be a positive integer")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Dataset task must be a nonempty string")
        if not math.isfinite(float(min_free_bytes)) or min_free_bytes < 0:
            raise ValueError("min_free_bytes must be finite and nonnegative")
        self.root = Path(root).expanduser().resolve()
        self.min_free_bytes = int(min_free_bytes)
        self.fps, self.episode_frames, self.task = float(fps), int(episode_frames), task
        self.capture_count = self.inference_count = self.episode_count = 0
        self.finalized_frames = 0
        self.finalized_inference_frames = 0
        self.quality = CaptureQuality()
        self._quality_warnings = 0
        self._capture_sequence = self._result_sequence = None
        self._capture_stamp = None
        self._first_stamp = None
        self._videos, self._video_stats = [], []
        self._segment_count = 0
        self._raw = None
        self._closed = False
        self._failure = None
        self._segment_failure = None
        self._writing_frame = False
        self.features = _features()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "meta").mkdir()
        (self.root / "raw").mkdir()
        self._capture_path = self.root / "raw/capture.jsonl"
        self._result_path = self.root / "raw/inference.jsonl"
        self._segment_path = self.root / "raw/segments.jsonl"
        self._capture = self._capture_path.open("w", encoding="utf-8", buffering=1)
        self._results = self._result_path.open("w", encoding="utf-8", buffering=1)
        self._segments = self._segment_path.open("w", encoding="utf-8", buffering=1)
        self.extra = dict(schema_version=1, recording_status="recording", profile=jsonable(profile),
                          fps=self.fps, task=self.task, episode_frames=self.episode_frames,
                          min_free_bytes=self.min_free_bytes,
                          cameras=jsonable(cameras), calibration_files={},
                          video_keys=list(VIDEO_KEYS), capture_count=0, inference_count=0,
                          semantics={
                              "timestamp": "episode-local constant-rate media time frame_index / fps; not sensor time",
                              "capture_time": "source timestamp minus first captured source timestamp, across episodes",
                              "receive_timestamp": "host monotonic clock seconds; retain source metadata for wall/device clocks",
                              "images": "lossless RGB H.264 of Source images before inference; source may apply demosaic/white balance",
                              "hands": "independent eye tracks; points are original input pixel XY, no stereo 3D triangulation",
                              "hand_vectors": "two slots per eye in JSON order; points reshape (2,21,2), visible (2,21); zero padding with present/valid masks",
                              "detections": "raw model fields retained in gesture_json; model world points are hand-local estimates, not global coordinates",
                              "inference_valid": "false means frame was not inferred, not a negative hand detection",
                              "raw_transport": "uncompressed original bytes; path/offset/length per frame, dtype/shape in capture_metadata_json",
                              "image_stats": "RGB [0,1] moments on spatial subsamples at most approximately 64x64 per frame",
                              "capture_scope": "all pairs delivered by Source.read; camera/ROS driver upstream losses cannot be recovered",
                          })
        try:
            self._snapshot(profile)
            self._update_extra()
        except BaseException:
            self._capture.close()
            self._results.close()
            self._segments.close()
            raise

    def _snapshot(self, profile):
        candidates = []
        base = Path(profile.get("_base", "."))
        for value in (profile.get("_path"), profile.get("calibration", {}).get("file")):
            if value:
                candidates.append(Path(value))
        if profile.get("camera"):
            camera_path = Path(profile["camera"])
            camera_path = camera_path if camera_path.is_absolute() else base / camera_path
            candidates.append(camera_path)
            if camera_path.is_file():
                import yaml
                bundle = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
                for key in ("calibration", "imu_calibration"):
                    if bundle.get(key):
                        path = Path(bundle[key])
                        candidates.append(path if path.is_absolute() else camera_path.parent / path)
        for channel in profile.get("channels", []):
            if channel.get("valid_mask"):
                candidates.append(Path(channel["valid_mask"]))
        destination = self.root / "meta/calibration"
        for index, source in enumerate(dict.fromkeys(path.resolve() for path in candidates)):
            if not source.is_file():
                raise FileNotFoundError(f"Recording calibration/profile asset is missing: {source}")
            destination.mkdir(exist_ok=True)
            target = destination / f"{index:02d}-{source.name}"
            shutil.copyfile(source, target)
            self.extra["calibration_files"][str(source)] = str(target.relative_to(self.root))
        model = Path(profile.get("gesture", {}).get("model", ""))
        if model.is_file():
            digest = hashlib.sha256()
            with model.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            self.extra["model"] = dict(path=str(model), sha256=digest.hexdigest())

    def _update_extra(self, status=None, error=None):
        if status:
            self.extra["recording_status"] = status
        self.extra.update(fps=self.fps, capture_count=self.capture_count, inference_count=self.inference_count,
                          episode_count=self.episode_count, finalized_frames=self.finalized_frames,
                          finalized_inference_frames=self.finalized_inference_frames,
                          unpublished_capture_frames=self.capture_count - self.finalized_frames,
                          unpublished_inference_frames=self.inference_count - self.finalized_inference_frames)
        if error:
            self.extra["error"] = str(error)
        _write_json(self.root / "meta/ego2.json", self.extra)

    def _start_segment(self, images):
        chunk, file = divmod(self.episode_count, 1000)
        self._segment_count = 0
        self._videos, self._video_stats = [], [_Moments(), _Moments()]
        for key, image in zip(VIDEO_KEYS, images):
            h, w, _ = image.shape
            if key not in self.features:
                self.features[key] = dict(dtype="video", shape=[h, w, 3], names=["height", "width", "channels"],
                    info={"video.height": h, "video.width": w, "video.codec": "h264", "video.pix_fmt": "gbrp",
                          "video.is_depth_map": False, "video.fps": self.fps, "video.channels": 3, "has_audio": False})
            elif self.features[key]["shape"] != list(image.shape):
                raise ValueError("Video resolution changed during capture")
            path = self.root / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
            self._videos.append(_Video(path, image.shape, self.fps, self.ffmpeg))

    def _finish_segment(self):
        if not self._videos:
            return
        errors = []
        for video in self._videos:
            try:
                video.close()
            except Exception as exc:
                errors.append(exc)
        if self._raw is not None:
            self._raw.close()
            self._raw = None
        self._videos = []
        if errors or self._segment_failure is not None:
            self._segment_failure = self._segment_failure or errors[0]
            raise self._segment_failure
        if not self._segment_count:
            return
        segment = dict(episode_index=self.episode_count, length=self._segment_count,
                       dataset_from_index=self.capture_count - self._segment_count,
                       dataset_to_index=self.capture_count,
                       video_stats={key: stats.stats(image=True, count=self._segment_count)
                                    for key, stats in zip(VIDEO_KEYS, self._video_stats)})
        self._segments.write(_json(segment) + "\n")
        self.episode_count += 1
        self._segment_count = 0

    def capture(self, pair):
        if self._closed:
            raise RuntimeError("Dataset is closed")
        try:
            self._capture_pair(pair)
        except BaseException as exc:
            self._failure = exc
            if self._writing_frame:
                # One eye may already contain the uncommitted frame. Never
                # publish this segment as a synchronized stereo pair stream.
                self._segment_failure = self._segment_failure or exc
            raise

    def _capture_pair(self, pair):
        sequence = int(pair.sequence)
        if self._capture_sequence is not None and sequence <= self._capture_sequence:
            raise ValueError("Capture source sequences must be strictly increasing")
        if not all(math.isfinite(float(x)) for x in (pair.stamp, pair.received)):
            raise ValueError("Capture timestamps must be finite")
        if self._capture_stamp is not None and float(pair.stamp) <= self._capture_stamp:
            raise ValueError("Capture source timestamps must be strictly increasing")
        if len(pair.images) != 2 or any(x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3 for x in pair.images):
            raise ValueError("Capture requires two uint8 BGR images")
        # Validate both images before either encoder receives the frame.
        for key, image in zip(VIDEO_KEYS, pair.images):
            if key in self.features and self.features[key]["shape"] != list(image.shape):
                raise ValueError("Video resolution changed during capture")
        raw = getattr(pair, "raw_transport", None)
        raw_size = raw.nbytes if isinstance(raw, np.ndarray) else len(raw) if raw is not None else 0
        # Reserve space for trailers/indexes instead of discovering ENOSPC
        # halfway through writing the two eyes. This runs on the writer worker.
        frame_budget = 2 * sum(image.nbytes for image in pair.images) + raw_size
        free = shutil.disk_usage(self.root).free
        if free < self.min_free_bytes + frame_budget:
            raise RuntimeError(f"Low disk space: {free / 1024**3:.2f} GiB free; "
                               f"stopping before the {self.min_free_bytes / 1024**3:.2f} GiB reserve is used")
        if self._segment_count == self.episode_frames:
            self._finish_segment()
        if not self._videos:
            self._start_segment(pair.images)
        if self._first_stamp is None:
            self._first_stamp = float(pair.stamp)
        metadata = jsonable(getattr(pair, "metadata", None) or {})
        self._writing_frame = True
        raw_path, raw_offset, raw_length = "", 0, 0
        if raw is not None:
            if isinstance(raw, np.ndarray):
                metadata["raw_transport"] = dict(shape=list(raw.shape), dtype=raw.dtype.str,
                                                 order="C", encoding="raw")
                payload = np.ascontiguousarray(raw).tobytes()
            else:
                payload = bytes(raw)
                metadata["raw_transport"] = dict(shape=[len(payload)], dtype="|u1", order="C", encoding="raw")
            chunk, file = divmod(self.episode_count, 1000)
            relative = Path(f"raw/transport/chunk-{chunk:03d}/file-{file:03d}.bin")
            if self._raw is None:
                (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
                self._raw = (self.root / relative).open("wb")
            raw_path, raw_offset, raw_length = str(relative), self._raw.tell(), len(payload)
            self._raw.write(payload)
        row = dict(sequence=sequence, source_timestamp=float(pair.stamp), receive_timestamp=float(pair.received),
                   capture_time=float(pair.stamp) - self._first_stamp, index=self.capture_count,
                   episode_index=self.episode_count, frame_index=self._segment_count,
                   metadata=metadata, raw_transport_path=raw_path,
                   raw_transport_offset=raw_offset, raw_transport_length=raw_length)
        for video, image, stats in zip(self._videos, pair.images, self._video_stats):
            video.write(image)
            step_y, step_x = max(1, math.ceil(image.shape[0] / 64)), max(1, math.ceil(image.shape[1] / 64))
            stats.add(image[::step_y, ::step_x, ::-1].reshape(-1, 3) / 255.0)
        self._capture.write(_json(row) + "\n")
        self.capture_count += 1
        self._segment_count += 1
        self._capture_sequence = sequence
        self._capture_stamp = float(pair.stamp)
        self._writing_frame = False
        for event in self.quality.observe(row):
            if event["stream"] == "host_receive" or self._quality_warnings >= 20:
                continue
            self._quality_warnings += 1
            print(f"[quality] {event['stream']} {event['kind']} at pair {sequence}: "
                  f"interval={event.get('interval_seconds', 0):.6f}s, "
                  f"estimated missing={event.get('estimated_missing_samples', 'unknown')}",
                  file=sys.stderr, flush=True)

    def result(self, pair, stamp, hands, detections=None, processing_ms=0.0):
        if self._closed:
            raise RuntimeError("Dataset is closed")
        try:
            self._result_pair(pair, stamp, hands, detections, processing_ms)
        except BaseException as exc:
            self._failure = exc
            raise

    def _result_pair(self, pair, stamp, hands, detections, processing_ms):
        sequence = int(pair.sequence)
        if self._result_sequence is not None and sequence <= self._result_sequence:
            raise ValueError("Inference source sequences must be strictly increasing")
        if not math.isfinite(float(stamp)) or not math.isfinite(float(processing_ms)) or processing_ms < 0:
            raise ValueError("Inference time and processing duration must be finite and nonnegative")
        if len(hands) != 2:
            raise ValueError("Inference result must contain two eye lists")
        serialized = jsonable(hands)
        _hand_values(serialized)  # Validate before committing the annotation journal.
        self._results.write(_json(dict(sequence=sequence, stamp=float(stamp), processing_ms=float(processing_ms),
                                       hands=serialized, detections=detections)) + "\n")
        self._result_sequence = sequence
        self.inference_count += 1

    def _schema(self):
        types = {"float32": self.pa.float32(), "float64": self.pa.float64(), "int64": self.pa.int64(),
                 "bool": self.pa.bool_(), "string": self.pa.string()}
        fields = []
        for name, feature in self.features.items():
            if feature["dtype"] == "video":
                continue
            dtype = types[feature["dtype"]]
            size = feature["shape"][0]
            fields.append(self.pa.field(name, dtype if size == 1 else self.pa.list_(dtype, size)))
        return self.pa.schema(fields)

    def _frame(self, capture, result):
        present = result is not None
        hands = result["hands"] if present else [[], []]
        row = dict(timestamp=capture["frame_index"] / self.fps, frame_index=capture["frame_index"],
                   episode_index=capture["episode_index"], index=capture["index"], task_index=0)
        for key in ("source_timestamp", "receive_timestamp", "capture_time", "raw_transport_path",
                    "raw_transport_offset", "raw_transport_length"):
            row["observation." + key] = capture[key]
        row.update({"observation.source_sequence": capture["sequence"], "observation.inference_valid": present,
                    "observation.inference_timestamp": result["stamp"] if present else 0.0,
                    "observation.processing_ms": result["processing_ms"] if present else 0.0,
                    "observation.gesture_json": _json(dict(hands=hands, detections=result["detections"] if present else None)),
                    "observation.capture_metadata_json": _json(capture["metadata"])})
        row.update(_hand_values(hands))
        return row

    def _finalize_tables(self, allow_unfinalized=False):
        schema = self._schema()
        captures, results = iter(_rows(self._capture_path)), iter(_rows(self._result_path))
        segments = _rows(self._segment_path)
        if allow_unfinalized:
            # A failed write may leave a partial JSONL tail after the last
            # committed row. Do not prefetch that tail while publishing the
            # verified prefix; retain it verbatim for diagnosis/recovery.
            results = islice(results, self.inference_count)
            segments = islice(segments, self.episode_count)
        annotation = next(results, None)
        global_stats = {}
        written = inferred = 0
        quality = CaptureQuality()
        for segment in segments:
            ep = segment["episode_index"]
            chunk, file = divmod(ep, 1000)
            path = self.root / DATA_PATH.format(chunk_index=chunk, file_index=file)
            path.parent.mkdir(parents=True, exist_ok=True)
            moments = {key: _Moments() for key, ft in self.features.items() if ft["dtype"] not in ("video", "string")}
            with self.pq.ParquetWriter(path, schema, compression="snappy") as writer:
                remaining = segment["length"]
                while remaining:
                    batch = []
                    for _ in range(min(128, remaining)):
                        capture = next(captures, None)
                        if capture is None or capture["episode_index"] != ep:
                            raise ValueError("Capture journal disagrees with finalized video segments")
                        if annotation is not None and annotation["sequence"] < capture["sequence"]:
                            raise ValueError("Inference journal references a frame that was not captured")
                        result = None
                        if annotation is not None and annotation["sequence"] == capture["sequence"]:
                            result, annotation = annotation, next(results, None)
                            inferred += 1
                        quality.observe(capture)
                        batch.append(self._frame(capture, result))
                    table = self.pa.Table.from_pylist(batch, schema=schema)
                    writer.write_table(table)
                    for key, accumulator in moments.items():
                        accumulator.add([row[key] for row in batch])
                    remaining -= len(batch)
                    written += len(batch)
            stats = {key: value.stats() for key, value in moments.items()}
            stats.update(segment["video_stats"])
            for key, feature_stats in stats.items():
                global_stats.setdefault(key, _Moments()).merge(feature_stats)
            episode = {key: value for key, value in segment.items() if key != "video_stats"}
            episode.update(tasks=[self.task], **{"data/chunk_index": chunk, "data/file_index": file,
                                               "meta/episodes/chunk_index": chunk, "meta/episodes/file_index": file})
            for key in VIDEO_KEYS:
                episode.update({f"videos/{key}/chunk_index": chunk, f"videos/{key}/file_index": file,
                                f"videos/{key}/from_timestamp": 0.0,
                                f"videos/{key}/to_timestamp": segment["length"] / self.fps})
            for key, values in stats.items():
                for stat, value in values.items():
                    episode[f"stats/{key}/{stat}"] = value
            metadata = self.root / EPISODES_PATH.format(chunk_index=chunk, file_index=file)
            metadata.parent.mkdir(parents=True, exist_ok=True)
            self.pq.write_table(self.pa.Table.from_pylist([episode]), metadata)
        if not allow_unfinalized and (
                written != self.capture_count or next(captures, None) is not None or annotation is not None):
            raise ValueError("Capture/inference journals contain unfinalized frames")
        # The pandas string index is part of the official tasks.parquet contract.
        self.pd.DataFrame({"task_index": [0]}, index=[self.task]).to_parquet(self.root / "meta/tasks.parquet")
        _write_json(self.root / "meta/stats.json", {key: value.stats(image=key in VIDEO_KEYS)
                                                    for key, value in global_stats.items()})
        info = dict(codebase_version="v3.0", robot_type="ego2_gesture", total_episodes=self.episode_count,
                    total_frames=written, total_tasks=1, chunks_size=1000, data_files_size_in_mb=100,
                    video_files_size_in_mb=200, fps=self.fps, splits={"train": f"0:{self.episode_count}"},
                    data_path=DATA_PATH, video_path=VIDEO_PATH, features=self.features)
        _write_json(self.root / "meta/info.json", info)
        self.finalized_frames = written
        self.finalized_inference_frames = inferred
        report = quality.summary()
        report["inference"] = dict(captured_frames=written, inferred_frames=inferred,
                                   uninferred_frames=written - inferred)
        report["scope"] = "Finalized dataset frames; raw journals may contain additional frames after a recording failure"
        sensor_issues = any(report[key]["available"] and any(report[key][field] for field in (
            "gap_count", "out_of_order_timestamps", "invalid_timestamps", "missing_timestamps"))
            for key in ("device_rgb", "imu"))
        sensor_issues |= bool(report["device_rgb"]["duplicate_timestamps"])
        sensor_issues |= bool(report["imu"]["available"] and report["imu"]["rows_without_accepted_imu"])
        report["sensor_continuity"] = ("issues_detected" if sensor_issues else "no_detected_gaps"
            if report["device_rgb"]["available"] and report["device_rgb"]["expected_hz"] is not None
            and report["imu"]["available"] else "unavailable")
        self.extra["quality"] = report
        _write_json(self.root / "meta/quality.json", report)

    def close(self, error=None):
        """Finalize after both producers stop; errors remain recorded in metadata."""
        if self._closed:
            return
        self._closed = True
        failure = error or self._failure
        encoder_failure = self._segment_failure
        try:
            try:
                self._finish_segment()
            except Exception as exc:
                encoder_failure = exc
            self._capture.close()
            self._results.close()
            self._segments.close()
            try:
                # Segment journal entries exist only after both encoders close
                # successfully. Publish that prefix while retaining all original
                # capture/annotation rows for diagnosis or later recovery.
                self._finalize_tables(allow_unfinalized=encoder_failure is not None or failure is not None)
            except Exception as exc:
                if encoder_failure is not None:
                    raise encoder_failure from exc
                raise
            if encoder_failure is not None:
                raise encoder_failure
        except BaseException as exc:
            self._update_extra("failed", encoder_failure or failure or exc)
            raise
        else:
            self._update_extra("failed" if failure else "complete" if self.capture_count else "empty", failure)
        finally:
            for handle in (self._capture, self._results, self._segments, self._raw):
                if handle is not None:
                    handle.close()
            for video in self._videos:
                try:
                    video.close()
                except Exception:
                    pass

    @classmethod
    def finalize_journals(cls, root, *, fps, task, features, extra, capture_count,
                          inference_count, episode_count):
        """Build tables from validated, closed journals and existing media.

        Recovery uses this entry point so it never reads today's profile/model
        assets or starts capture encoders. Callers must validate all media and
        reconstruct the segment journal before publishing a complete dataset.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd

        writer = cls.__new__(cls)
        writer.pa, writer.pq, writer.pd = pa, pq, pd
        writer.root, writer.fps, writer.task = Path(root), float(fps), task
        writer.features, writer.extra = features, extra
        writer.capture_count, writer.inference_count = capture_count, inference_count
        writer.episode_count, writer.finalized_frames = episode_count, 0
        writer._capture_path = writer.root / "raw/capture.jsonl"
        writer._result_path = writer.root / "raw/inference.jsonl"
        writer._segment_path = writer.root / "raw/segments.jsonl"
        writer._finalize_tables()
        writer._update_extra("complete")

    finalize = close
