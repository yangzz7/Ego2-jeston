"""Stream recorded LeRobot v3 gesture episodes without camera or model access."""
from contextlib import ExitStack, closing
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from ..common.sources import Pair
from .tracking import Hand


VIDEO_KEYS = ("observation.images.eye0", "observation.images.eye1")
FRAME_FIELDS = (
    "index", "episode_index", "frame_index", "timestamp",
    "observation.source_timestamp", "observation.receive_timestamp",
    "observation.capture_time", "observation.source_sequence",
    "observation.inference_valid", "observation.gesture_json", "observation.capture_metadata_json",
)


def _json_file(path):
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read dataset metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Dataset metadata must be a JSON object: {path}")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"Invalid {name}: expected an integer >= {minimum}, got {value!r}")
    return int(value)


def _finite(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: expected a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"Invalid {name}: expected a finite number")
    return result


def _hands(raw, valid):
    try:
        result = json.loads(raw)
        eyes = result["hands"]
        if not isinstance(eyes, list) or len(eyes) != 2:
            raise ValueError("hands must contain two eye lists")
        output = []
        for eye in eyes:
            if not isinstance(eye, list):
                raise ValueError("each eye must contain a hand list")
            hands = []
            for saved in eye:
                points = np.asarray(saved["points"], dtype=np.float64)
                visible = np.asarray(saved["visible"], dtype=bool)
                if points.shape != (21, 2) or visible.shape != (21,):
                    raise ValueError("hand points/visible must have shapes (21, 2)/(21,)")
                if not np.isfinite(points[visible]).all():
                    raise ValueError("visible hand points must be finite")
                values = dict(saved, points=points, visible=visible)
                _integer(values["id"], "hand id")
                for name in ("handedness", "gesture", "pinch"):
                    if not isinstance(values[name], str):
                        raise ValueError(f"hand {name} must be a string")
                for name in ("score", "pinch_ratio"):
                    if values[name] is not None:
                        values[name] = _finite(values[name], f"hand {name}")
                hands.append(Hand(**values))
            output.append(hands)
        if not valid and any(output):
            raise ValueError("a frame without inference cannot contain hands")
        return output
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid observation.gesture_json: {exc}") from exc


class _Video:
    """Decode in order, including a shared file's prefix, to avoid keyframe seeks."""

    def __init__(self, path, fps, start, length, shape):
        if not path.is_file():
            raise ValueError(f"Missing dataset video: {path}")
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        self.shape = shape
        try:
            if not self.capture.isOpened():
                raise ValueError(f"Cannot decode dataset video: {path}")
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(actual_fps) or abs(actual_fps - fps) > max(0.001, fps * 0.001):
                raise ValueError(f"Video fps does not match metadata ({actual_fps} != {fps}): {path}")
            reported_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if math.isfinite(reported_count) and reported_count > 0 and reported_count < start + length:
                raise ValueError(f"Video frame count {reported_count:g} is shorter than expected "
                                 f"{start + length}: {path}")
            self.position = 0
            for _ in range(start):
                self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        ok, image = self.capture.read()
        if not ok or image is None:
            raise ValueError(f"Cannot decode video frame {self.position} (truncated/corrupt video): {self.path}")
        self.position += 1
        if tuple(image.shape) != self.shape:
            raise ValueError(f"Video shape {image.shape} differs from metadata {self.shape}: {self.path}")
        return image

    def close(self):
        self.capture.release()


class DatasetReplay:
    """Read bounded Parquet batches and one stereo image pair at a time."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("LeRobot replay requires pyarrow; install requirements-dataset.txt") from exc
        self.parquet = parquet
        self.info = _json_file(self.root / "meta/info.json")
        self.metadata = _json_file(self.root / "meta/ego2.json")
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("Replay requires a LeRobot v3.0 dataset (meta/info.json codebase_version)")
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Unsupported or missing Ego2 metadata schema_version in meta/ego2.json")
        self.profile = self.metadata.get("profile")
        if not isinstance(self.profile, dict):
            raise ValueError("Missing saved profile in meta/ego2.json")
        channels = self.profile.get("channels")
        if (not isinstance(channels, list) or len(channels) != 2 or
                any(not isinstance(channel, dict) or not channel.get("camera") for channel in channels)):
            raise ValueError("Saved profile must describe exactly two camera channels")
        # Old/minimal recording profiles need only this display preference.
        self.profile.setdefault("viewer", {}).setdefault("memory_limit", "2GB")
        self.fps = _finite(self.info.get("fps"), "dataset fps")
        if self.fps <= 0:
            raise ValueError("Dataset fps must be positive")
        self.total_episodes = _integer(self.info.get("total_episodes"), "total_episodes")
        self.total_frames = _integer(self.info.get("total_frames"), "total_frames")
        if not self.total_episodes or not self.total_frames:
            raise ValueError("Dataset contains no recorded episodes/frames")
        self.shapes = []
        features = self.info.get("features", {})
        for key in VIDEO_KEYS:
            feature = features.get(key, {})
            shape = feature.get("shape")
            if (feature.get("dtype") != "video" or not isinstance(shape, list) or len(shape) != 3 or
                    any(not isinstance(n, int) or n <= 0 for n in shape) or shape[-1] != 3):
                raise ValueError(f"Missing/invalid video feature metadata: {key}")
            self.shapes.append(tuple(shape))
        if self.metadata.get("recording_status") not in ("complete",):
            print(f"[replay] Warning: recording status is {self.metadata.get('recording_status')!r}; "
                  "replaying available finalized episodes", file=sys.stderr, flush=True)

    def _path(self, template, **values):
        try:
            path = (self.root / self.info[template].format(**values)).resolve()
        except (KeyError, IndexError, AttributeError, ValueError) as exc:
            raise ValueError(f"Invalid {template} template in meta/info.json") from exc
        if not path.is_relative_to(self.root):
            raise ValueError(f"Dataset {template} must remain inside the dataset directory")
        return path

    def _rows(self, path):
        try:
            source = self.parquet.ParquetFile(path)
            try:
                for batch in source.iter_batches(batch_size=128):
                    yield from batch.to_pylist()
            finally:
                source.close()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read dataset Parquet file {path}: {exc}") from exc

    def _episodes(self, selected):
        files = sorted((self.root / "meta/episodes").glob("chunk-*/file-*.parquet"))
        if not files:
            raise ValueError("Missing v3 episode metadata: meta/episodes/chunk-*/file-*.parquet")
        previous, found, count = -1, False, 0
        for path in files:
            for episode in self._rows(path):
                number = _integer(episode.get("episode_index"), "episode_index")
                if number <= previous:
                    raise ValueError("Episode metadata is duplicated or out of order")
                previous = number
                count += 1
                if selected is None or selected == number:
                    found = True
                    yield episode
        if selected is not None and not found:
            raise ValueError(f"Episode {selected} does not exist in {self.root}")
        if count != self.total_episodes:
            raise ValueError(f"Episode metadata count {count} differs from total_episodes {self.total_episodes}")

    def frames(self, episode=None):
        """Yield (Pair, original relative capture time, restored hands, saved row)."""
        if episode is not None:
            _integer(episode, "selected episode")
        previous_time, count = -math.inf, 0
        for saved in self._episodes(episode):
            try:
                number = _integer(saved["episode_index"], "episode_index")
                length = _integer(saved["length"], f"episode {number} length", 1)
                first = _integer(saved["dataset_from_index"], "dataset_from_index")
                last = _integer(saved["dataset_to_index"], "dataset_to_index", 1)
                if last - first != length:
                    raise ValueError(f"Episode {number} index range does not match its length")
                data_path = self._path("data_path", chunk_index=saved["data/chunk_index"],
                                       file_index=saved["data/file_index"])
                with ExitStack() as resources:
                    videos = []
                    for key, shape in zip(VIDEO_KEYS, self.shapes):
                        prefix = f"videos/{key}/"
                        start = _finite(saved[prefix + "from_timestamp"], f"{key} from_timestamp") * self.fps
                        end = _finite(saved[prefix + "to_timestamp"], f"{key} to_timestamp") * self.fps
                        if start < 0 or abs(start - round(start)) > 0.01 or abs(end - start - length) > 0.01:
                            raise ValueError(f"Episode {number} video offsets do not match frame count: {key}")
                        path = self._path("video_path", video_key=key,
                                          chunk_index=saved[prefix + "chunk_index"],
                                          file_index=saved[prefix + "file_index"])
                        video = _Video(path, self.fps, round(start), length, shape)
                        resources.callback(video.close)
                        videos.append(video)
                    offset = 0
                    rows = resources.enter_context(closing(self._rows(data_path)))
                    for row in rows:
                        index = _integer(row.get("index"), "frame index")
                        if index < first or index >= last:
                            continue
                        missing = [name for name in FRAME_FIELDS if name not in row]
                        if missing:
                            raise ValueError(f"Episode {number} is missing frame fields: {', '.join(missing)}")
                        if index != first + offset or row["episode_index"] != number or row["frame_index"] != offset:
                            raise ValueError(f"Episode {number} frame indices are missing, duplicated or out of order")
                        timestamp = _finite(row["timestamp"], "timestamp")
                        tolerance = max(1e-4, 0.01 / self.fps, abs(timestamp) * np.finfo(np.float32).eps * 2)
                        if abs(timestamp - offset / self.fps) > tolerance:
                            raise ValueError(f"Episode {number} frame {offset} timestamp does not match its video frame")
                        stamp = _finite(row["observation.capture_time"], "observation.capture_time")
                        if stamp < 0 or stamp < previous_time:
                            raise ValueError("Recorded capture_time is negative or goes backwards")
                        previous_time = stamp
                        inferred = row["observation.inference_valid"]
                        if not isinstance(inferred, bool):
                            raise ValueError("observation.inference_valid must be boolean")
                        hands = _hands(row["observation.gesture_json"], inferred)
                        try:
                            capture_metadata = json.loads(row["observation.capture_metadata_json"])
                        except (TypeError, ValueError) as exc:
                            raise ValueError("Invalid observation.capture_metadata_json") from exc
                        if not isinstance(capture_metadata, dict):
                            raise ValueError("observation.capture_metadata_json must contain an object")
                        pair = Pair(
                            tuple(video.read() for video in videos),
                            _finite(row["observation.source_timestamp"], "observation.source_timestamp"),
                            _finite(row["observation.receive_timestamp"], "observation.receive_timestamp"),
                            _integer(row["observation.source_sequence"], "observation.source_sequence"),
                            metadata=capture_metadata,
                        )
                        offset += 1
                        count += 1
                        yield pair, stamp, hands, row
                    if offset != length:
                        raise ValueError(f"Episode {number} has {offset} data frames; expected {length}")
            except KeyError as exc:
                raise ValueError(f"Missing required episode metadata: {exc.args[0]}") from exc
        if episode is None and count != self.total_frames:
            raise ValueError(f"Dataset contains {count} frames; meta/info.json declares {self.total_frames}")


def replay(args):
    """Send saved imagery, hand overlays, timestamps and metadata to Rerun."""
    dataset = DatasetReplay(args.replay)
    # A capture's live-preview ring budget must not silently trim offline replay.
    dataset.profile["viewer"]["memory_limit"] = getattr(args, "viewer_memory", None) or "50%"
    if getattr(args, "rrd_lossless", False):
        dataset.profile["viewer"]["jpeg_quality"] = 0
    count, inferred = 0, 0
    started = time.monotonic()
    first_stamp = None
    with ExitStack() as resources:
        from .visualizer import Visualizer
        viewer = Visualizer(dataset.profile, headless=args.headless, save=args.save,
                            connect=args.connect, web=args.web, web_port=args.web_port,
                            grpc_port=args.grpc_port)
        resources.callback(viewer.close)
        viewer.log_session(dataset.metadata)
        frames = resources.enter_context(closing(dataset.frames(args.episode)))
        print(f"[replay] dataset={dataset.root} episode={args.episode if args.episode is not None else 'all'}", flush=True)
        try:
            while not args.frames or count < args.frames:
                if args.seconds and time.monotonic() - started >= args.seconds:
                    break
                try:
                    pair, stamp, hands, row = next(frames)
                except StopIteration:
                    break
                if first_stamp is None:
                    first_stamp = stamp
                    started = time.monotonic()
                if args.realtime:
                    delay = stamp - first_stamp - (time.monotonic() - started)
                    if delay > 0:
                        if args.seconds and time.monotonic() - started + delay >= args.seconds:
                            break
                        time.sleep(delay)
                valid = row["observation.inference_valid"]
                viewer.log(pair, stamp, hands, inference_ran=valid)
                viewer.log_frame_metadata(row)
                count += 1
                inferred += valid
            if count == 0:
                raise ValueError("No frames were replayed")
            viewer.recording.flush(timeout_sec=5)
            print(f"[replay] replayed={count} inferred={inferred} missing_inference={count - inferred}", flush=True)
            if viewer.web_process is not None:
                print("[replay] Browser replay is ready; press Ctrl+C to stop the local viewer server.", flush=True)
                while viewer.web_process.poll() is None:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[replay] Stopping", flush=True)
    metrics = dict(replayed_pairs=count, inference_pairs=inferred, missing_inference_pairs=count - inferred)
    if args.metrics:
        path = Path(args.metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
