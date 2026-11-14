"""Bounded live capture, paired inference and deterministic offline playback."""
from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import sys
import signal

import numpy as np

from ..common.sources import Source
from .recognizer import Recognizer
from .rectification import Rectifier
from .tracking import HandTracker


from ..common.runtime import FrameClock, LatestPairs, QueuedArchive


class _CaptureResources(ExitStack):
    """Finish the current frame on Ctrl+C and protect archive finalization."""
    def __init__(self, stop=None):
        super().__init__()
        self.stop = stop
        self._entry_sigint = None

    def __enter__(self):
        super().__enter__()
        if self.stop is not None and threading.current_thread() is threading.main_thread():
            self._entry_sigint = signal.getsignal(signal.SIGINT)
            # Raising KeyboardInterrupt in a JSONL write can leave half a
            # record. Ask the loop to stop after its current commit instead.
            signal.signal(signal.SIGINT, lambda signum, frame: self.stop.set())
        return self

    def __exit__(self, *error):
        main_thread = threading.current_thread() is threading.main_thread()
        previous = signal.getsignal(signal.SIGINT) if main_thread else None
        if main_thread:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            return super().__exit__(*error)
        finally:
            if main_thread:
                signal.signal(signal.SIGINT, self._entry_sigint if self._entry_sigint is not None else previous)


class Pipeline:
    def __init__(self, profile, cameras):
        self.profile = profile
        self.clock = FrameClock()
        self.rectifiers = [Rectifier(cameras[c["camera"]], c, profile["inference_view"])
                           for c in profile["channels"]]
        aspect = profile["inference_view"]["width"] / profile["inference_view"]["height"]
        self.trackers = [HandTracker(profile["gesture"], aspect) for _ in range(2)]
        self.recognizers = []
        self.inference_status = []
        self.observations = [[], []]
        self.tracking_diagnostics = []
        try:
            self.recognizers = [Recognizer(profile["gesture"])]
            self.recognizers.append(Recognizer(profile["gesture"]))
        except Exception:
            self.close()
            raise

    def process(self, pair):
        if len(pair.images) != 2:
            raise ValueError("Expected exactly two images per stereo frame")
        stamp, timestamp_ms = self.clock.update(pair.stamp)
        results = []
        self.observations = []
        self.tracking_diagnostics = []
        for image, rectifier, recognizer, tracker in zip(
                pair.images, self.rectifiers, self.recognizers, self.trackers):
            rgb = rectifier.prepare(image)
            detections, visibility, observations, accepted = [], [], [], []
            for detection in recognizer.detect(rgb, timestamp_ms):
                projected, valid = rectifier.backproject(detection.points)
                observation = dict(
                    points=detection.points, world_points=detection.world_points,
                    handedness=detection.handedness, handedness_score=detection.handedness_score,
                    gesture=detection.gesture, score=detection.score,
                    handedness_categories=detection.handedness_categories,
                    gesture_categories=detection.gesture_categories,
                    backprojected_points=projected, visible=valid, accepted=False,
                    model_timestamp_ms=timestamp_ms,
                )
                observations.append(observation)
                if valid[0] and valid.sum() >= 10:
                    detections.append(detection)
                    visibility.append(valid)
                    observation["accepted"] = True
                    accepted.append(observation)
            hands = tracker.update(detections, stamp, visibility)
            mapped = []
            for hand, observation in zip(hands, accepted):
                points, valid = rectifier.backproject(hand.points)
                observation.update(track_id=hand.id, smoothed_points=hand.points.copy())
                mapped.append(replace(hand, points=points, visible=valid & hand.visible))
            results.append(mapped)
            self.observations.append(observations)
            self.tracking_diagnostics.append(dict(events=list(tracker.events), stats=dict(tracker.stats)))
        return stamp, results

    def close(self):
        if self.recognizers:
            self.inference_status = [
                dict(backend=getattr(r, "backend", "unknown"),
                     requested=getattr(r, "requested_backend", "unknown"),
                     fallbacks=getattr(r, "fallbacks", 0))
                for r in self.recognizers]
        for recognizer in self.recognizers:
            recognizer.close()
        self.recognizers.clear()
        for rectifier in self.rectifiers:
            rectifier.close()


def run(profile, args, *, viewer_factory=None, stop_event=None):
    stop = stop_event if stop_event is not None else threading.Event()
    durations, ages = [], []
    count = 0
    hand_frames = [0, 0]
    capture = None
    archive_worker = None
    dataset = None
    started = None
    viewer_errors = []

    def viewer_failed(stage, error):
        # Preview transport is optional. In particular, a closed Rerun window
        # must not stop the camera, inference, or dataset finalization.
        viewer_errors.append(dict(stage=stage, type=type(error).__name__, message=str(error)))
        consequence = "dataset finalization continues" if stage == "close" else "recording continues without live preview"
        print(f"[viewer] {stage} failed: {error}; {consequence}",
              file=sys.stderr, flush=True)

    def close_viewer(active_viewer):
        try:
            active_viewer.close()
        except Exception as exc:
            viewer_failed("close", exc)

    clock_kind = ("host_receive" if profile["input"]["type"] in ("cyper", "uvc") else
                  "ros_header" if profile["input"]["type"] == "ros2" else "media_time")
    with _CaptureResources(stop) as resources:
        pipeline = Pipeline(profile, args.cameras)
        resources.callback(pipeline.close)
        inference_reporter = getattr(args, 'inference_reporter', None)
        if inference_reporter:
            inference_reporter(pipeline.recognizers)
        source = Source(profile, args.cameras)
        if source.kind == "cyper":
            from .preprocess import WhiteBalance
            balance = WhiteBalance()
            source.white_balance_processor = balance
            # Same per-frame pixels; keep IMU capture from waiting between
            # RAW conversion and four separate white-balance GPU submissions.
            source.fuse_white_balance = balance.accelerator.mode == 'cuda'
            if source.fuse_white_balance:
                # One buffer can be read out while the previous pair is processed.
                source.capture_buffer_count = 2
            resources.callback(balance.close)
        live = source.kind in ("cyper", "uvc", "ros2")
        if not live:
            resources.callback(source.close)
            source.open()
        if getattr(args, "dataset", None) is not None:
            from .dataset import GestureDataset
            dataset = GestureDataset(args.dataset, profile, args.cameras,
                                     fps=args.dataset_fps or source.fps, task=args.task,
                                     episode_frames=args.episode_frames,
                                     min_free_bytes=int(getattr(args, "min_free_gb", 2.0) * 1024**3))
            def finalize_dataset(exc_type, exc, traceback):
                if capture is not None and capture.thread.is_alive():
                    raise RuntimeError("Capture worker is still active; dataset was left incomplete")
                if archive_worker is not None and archive_worker.thread.is_alive():
                    raise RuntimeError("Archive worker is still active; dataset was left incomplete")
                failure = exc or (capture.error if capture else None) or (archive_worker.error if archive_worker else None)
                if archive_worker is not None:
                    dataset.extra["archive_peak_pending"] = archive_worker.peak_pending
                    dataset.extra["archive_queue_capacity"] = archive_worker.pending.maxsize
                if viewer_errors:
                    dataset.extra["viewer_errors"] = viewer_errors
                dataset.close(error=str(failure) if failure is not None else None)
                return False
            resources.push(finalize_dataset)
            print(f"[dataset] LeRobot v3.0: {args.dataset}", flush=True)
        manual = None
        if getattr(args, 'recording_control', None):
            from ..multi_vio.gesture_recording import LiveGestureRecording
            manual = LiveGestureRecording(args.recording_control,profile,args,source)
            def finalize_manual(exc_type, exc, traceback):
                failure = exc or (capture.error if capture is not None else None)
                manual.close(error=str(failure) if failure is not None else None)
                return False
            resources.push(finalize_manual)
        viewer = None
        if not args.no_viz:
            try:
                from .visualizer import Visualizer
                viewer = (viewer_factory or Visualizer)(profile, headless=args.headless, save=args.save, connect=args.connect,
                                    web=args.web, web_port=args.web_port, grpc_port=args.grpc_port)
                resources.callback(close_viewer, viewer)
                viewer.log_session(dict(profile=profile, clock_source=clock_kind,
                                        note="Live inference preview; dataset replay includes every archived pair."))
            except Exception as exc:
                viewer_failed("startup", exc)
                viewer = None
        if live:
            def archive(pair):
                # Source.open runs on the capture worker and may negotiate a
                # different frame rate. No video has been opened at this point.
                if dataset.capture_count == 0 and args.dataset_fps is None:
                    dataset.fps = source.fps
                dataset.capture(pair)
            if dataset is not None:
                # Encoder rollover and disk writes must not hold up the camera
                # reader. Drain every queued pair before finalizing the dataset.
                archive_worker = QueuedArchive(archive).start()
                resources.callback(archive_worker.close)
            archive_pair = manual.capture if manual else archive_worker.submit if archive_worker else None
            imu_observer = viewer.capture_imu if getattr(viewer, 'imu_publisher', None) is not None else None
            on_pair = archive_pair
            if imu_observer is not None:
                def on_pair(pair):
                    # Durable recording keeps its existing callback and order.
                    # The observer only batches small scalar values; all source
                    # IMU samples precede the lossy inference-frame slot.
                    if archive_pair is not None:
                        archive_pair(pair)
                    imu_observer(pair)
            capture = LatestPairs(source, stop, on_pair=on_pair)
            resources.callback(capture.close)
            capture.start()
            read = capture.read
        else:
            read = lambda: source.read(stop)
        started = last_report = time.monotonic()
        print(f"[gesture] source={source.kind} clock={clock_kind}; independent left/right tracking", flush=True)
        try:
            while not stop.is_set():
                if manual: manual.poll()
                if args.frames and count >= args.frames:
                    break
                if args.seconds and time.monotonic() - started >= args.seconds:
                    break
                pair = read()
                if pair is None:
                    break
                if dataset is not None and not live:
                    dataset.capture(pair)
                if manual and not live: manual.capture(pair)
                begin = time.monotonic()
                stamp, hands = pipeline.process(pair)
                processing_ms = (time.monotonic() - begin) * 1000
                if inference_reporter:
                    inference_reporter(pipeline.recognizers)
                if dataset is not None:
                    dataset.result(pair, stamp, hands, pipeline.observations, processing_ms)
                if manual: manual.result(pair, stamp, hands, pipeline.observations, processing_ms)
                if viewer is not None:
                    try:
                        if hasattr(viewer, 'log_observations'):
                            if hasattr(viewer, 'set_tracking_diagnostics'):
                                viewer.set_tracking_diagnostics(getattr(pipeline, 'tracking_diagnostics', []))
                            viewer.log_observations(pair, stamp, hands, pipeline.observations)
                        else:
                            viewer.log(pair, stamp, hands)
                        viewer.log_frame_metadata({
                            "observation.capture_time": stamp,
                            "observation.source_timestamp": pair.stamp,
                            "observation.source_sequence": pair.sequence,
                            "observation.inference_valid": True,
                            "observation.processing_ms": processing_ms,
                            "observation.capture_metadata_json": pair.metadata,
                        })
                    except Exception as exc:
                        viewer_failed("log", exc)
                        viewer = None
                end = time.monotonic()
                durations.append((end - begin) * 1000)
                ages.append((end - pair.received) * 1000)
                # Bounded diagnostics even for an indefinitely running camera.
                if len(durations) > 1000:
                    del durations[:500]
                    del ages[:500]
                count += 1
                for eye in range(2):
                    hand_frames[eye] += bool(hands[eye])
                if end - last_report >= 2:
                    print(f"[gesture] pairs={count} rate={count/(end-started):.1f}/s "
                          f"process={np.mean(durations[-60:]):.1f} ms "
                          f"hands={len(hands[0])}/{len(hands[1])} "
                          f"dropped={capture.dropped if capture else 0}", flush=True)
                    last_report = end
        except KeyboardInterrupt:
            print("\n[gesture] Stopping", flush=True)
        finally:
            stop.set()
            if dataset is not None:
                print("[dataset] Draining archive queue and finalizing videos/tables; please wait...", flush=True)
        elapsed = time.monotonic() - started
    if count == 0:
        raise RuntimeError("No stereo frames were processed")
    completed_inference = None
    if dataset is not None and getattr(args, "complete_inference", False):
        from .reinfer import reinfer_dataset
        print("[dataset] Re-running inference on every archived frame; live results will be preserved...", flush=True)
        completed_inference = reinfer_dataset(args.dataset)
    metrics = dict(
        processed_pairs=count, hand_frames=hand_frames, elapsed_seconds=elapsed,
        output_pairs_per_second=count / max(elapsed, 1e-9),
        processing_ms_mean=float(np.mean(durations)), processing_ms_p95=float(np.percentile(durations, 95)),
        receive_to_log_ms_mean=float(np.mean(ages)),
        dropped_pairs=capture.dropped if capture else 0, clock_source=clock_kind,
        archive_peak_pending=archive_worker.peak_pending if archive_worker else 0,
        viewer_errors=viewer_errors,
        inference_backends=getattr(pipeline, "inference_status", []),
        note="dropped_pairs counts skipped inference pairs, not dataset loss. Dataset archives every pair returned by Source before inference selection. Timing samples cover at most the last 1000 pairs; receive_to_log excludes sensor and viewer latency.",
    )
    if dataset is not None:
        metadata = json.loads((Path(args.dataset) / "meta" / "info.json").read_text())
        metrics.update(dataset=str(args.dataset), captured_pairs=metadata["total_frames"],
                       uninferred_pairs=metadata["total_frames"] - count)
        if completed_inference is not None:
            metrics["live_uninferred_pairs"] = metrics["uninferred_pairs"]
            metrics["uninferred_pairs"] = 0
            metrics["completed_inference"] = completed_inference
        quality_path = Path(args.dataset) / "meta/quality.json"
        if quality_path.is_file():
            quality = json.loads(quality_path.read_text())
            metrics["quality"] = quality
            rgb, imu = quality.get("device_rgb", {}), quality.get("imu", {})
            uninferred = quality.get("inference", {}).get("uninferred_frames", metrics["uninferred_pairs"])
            print(f"[quality] sensors={quality.get('sensor_continuity', 'unavailable')}; "
                  f"RGB gaps={rgb.get('gap_count', 0)} "
                  f"(estimated missing pairs={rgb.get('estimated_missing_samples', 0)}); "
                  f"IMU gaps={imu.get('gap_count', 0)} "
                  f"(max interval={1000 * (imu.get('max_interval_seconds') or 0):.1f} ms); "
                  f"uninferred frames={uninferred}", flush=True)
    if args.metrics:
        path = Path(args.metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if dataset is not None and args.save and not args.no_viz:
        from .replay import replay
        destination = Path(args.save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        print("[dataset] Exporting complete Rerun replay (all captured frames)...", flush=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".rrd", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            exported = replay(SimpleNamespace(**{**vars(args), "replay": args.dataset, "save": temporary,
                                      "headless": True, "connect": None, "web": False,
                                      "frames": 0, "seconds": 0, "episode": None, "realtime": False,
                                      "metrics": None}))
            if exported["replayed_pairs"] != metrics["captured_pairs"]:
                raise RuntimeError("Rerun export interrupted; dataset is complete, but the RRD preview was not replaced. Run --replay --headless --save to export again.")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
