"""Launch from the repository root: python3 -m ego2.gesture."""
import argparse
from pathlib import Path
import math
import sys
from datetime import datetime

from .config import DEFAULT_CONFIG, load_config
from ..common.camera import add_camera_argument, validate_camera_args, print_camera_selection


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cyperstereo fisheye hand tracking + Rerun four-view display")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_camera_argument(parser)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--device", help="Explicit V4L2 device; overrides serial auto-discovery")
    source.add_argument("--demo", action="store_true", help="Synthetic geometry scene; real model runs, no simulated hands")
    source.add_argument("--images", nargs=2, metavar=("LEFT", "RIGHT"), help="Two calibrated fisheye images")
    source.add_argument("--video", help="Side-by-side stereo video, processed in order without dropping frames")
    source.add_argument("--ros2", nargs=2, metavar=("LEFT_TOPIC", "RIGHT_TOPIC"))
    source.add_argument("--replay", type=Path, help="Replay an Ego2 LeRobot V3 dataset without a camera or model")
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument("--dataset", type=Path, help="New LeRobot V3 output directory (default: output/gesture/<timestamp>)")
    recording.add_argument("--no-record", action="store_true", help="Disable LeRobot capture and automatic viewer.rrd")
    parser.add_argument("--task", default="Stereo hand gesture capture", help="Natural-language dataset task description")
    parser.add_argument("--dataset-fps", type=float, help="Nominal LeRobot/video FPS; original capture times are also saved")
    parser.add_argument("--episode-frames", type=int, default=900, help="Start a new dataset episode after this many captured pairs")
    parser.add_argument("--no-raw", action="store_true", help="Omit packed Cyper Bayer transport; keep RGB, IMU and metadata")
    parser.add_argument("--complete-inference", action="store_true", help="After recording, re-run every saved frame in order to eliminate skipped inference frames")
    parser.add_argument("--preflight", action="store_true", help="Check recording readiness and exit without opening the camera or creating the dataset")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="Stop recording before using this disk reserve in GiB (default: 2)")
    parser.add_argument("--episode", type=int, help="Replay only this zero-based episode")
    parser.add_argument("--realtime", action="store_true", help="Pace dataset replay using original capture times")
    parser.add_argument("--model", type=Path, help="Override local .task model path")
    parser.add_argument("--download-model", action="store_true", help="Download/verify official model and exit")
    parser.add_argument("--hfov", type=float, help="Internal perspective horizontal field of view")
    parser.add_argument("--pitch", type=float, help="Internal view pitch; negative looks down")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N processed pairs (0: unlimited)")
    parser.add_argument("--seconds", type=float, default=0, help="Stop after this duration (0: unlimited)")
    parser.add_argument("--headless", action="store_true", help="Record without launching a Rerun window")
    parser.add_argument("--no-viz", action="store_true", help="Run capture/inference without Rerun")
    parser.add_argument("--save", type=Path, help="Save synchronized images and annotations to .rrd")
    parser.add_argument("--connect", help="Connect to an existing Rerun viewer (rerun+http://host:9876/proxy)")
    parser.add_argument("--web", action="store_true", help="Use local browser Rerun; automatic on Linux without a desktop display")
    parser.add_argument("--web-port", type=int, default=9090, help="Local browser server HTTP port")
    parser.add_argument("--grpc-port", type=int, default=9876, help="Local browser server gRPC port")
    parser.add_argument("--viewer-memory", help="Rerun memory budget, e.g. 4GiB or 50%%; replay defaults to 50%%")
    parser.add_argument("--rrd-lossless", action="store_true", help="Store uncompressed RGB in RRD (large); dataset videos always remain lossless")
    parser.add_argument("--metrics", type=Path, help="Write final performance metrics to JSON")
    args = parser.parse_args(argv)
    validate_camera_args(parser, args)
    if args.frames < 0 or not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--frames and --seconds must be nonnegative and finite")
    if args.episode_frames <= 0:
        parser.error("--episode-frames must be positive")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        parser.error("--min-free-gb must be nonnegative and finite")
    if args.preflight and (args.no_record or args.replay or args.download_model):
        parser.error("--preflight requires a new recording dataset")
    if args.dataset_fps is not None and (not math.isfinite(args.dataset_fps) or args.dataset_fps <= 0):
        parser.error("--dataset-fps must be positive and finite")
    if not args.task.strip():
        parser.error("--task must not be empty")
    if args.episode is not None and (not args.replay or args.episode < 0):
        parser.error("--episode requires --replay and a nonnegative episode index")
    if args.realtime and not args.replay:
        parser.error("--realtime requires --replay")
    if args.replay and (args.dataset or args.no_viz or args.download_model):
        parser.error("--replay cannot be combined with --dataset, --no-viz or --download-model")
    if args.complete_inference and (args.no_record or args.replay):
        parser.error("--complete-inference requires recording a dataset")
    if args.no_viz and (args.save or args.connect or args.web):
        parser.error("--no-viz cannot be combined with --save, --connect or --web")
    if args.web and args.connect:
        parser.error("--web starts a server; use --connect separately for an existing viewer")
    if not all(1 <= p <= 65535 for p in (args.web_port, args.grpc_port)) or args.web_port == args.grpc_port:
        parser.error("--web-port and --grpc-port must be distinct ports in 1..65535")
    if args.headless and (args.replay or args.no_record) and not (args.save or args.connect or args.web or args.no_viz or args.download_model):
        parser.error("--headless requires --save, --connect, --web or --no-viz")
    try:
        if args.replay:
            from .replay import replay
            replay(args)
            return 0
        profile, args.cameras = load_config(args.config, sn=args.sn)
        print_camera_selection(profile)
        if args.viewer_memory:
            profile["viewer"]["memory_limit"] = args.viewer_memory
        if args.rrd_lossless:
            profile["viewer"]["jpeg_quality"] = 0
        if args.model:
            profile["gesture"]["model"] = str(args.model.expanduser().resolve())
        if args.download_model:
            from .model import download_model
            print(download_model(profile["gesture"]["model"]))
            return 0
        if args.device:
            profile["input"].update(type="cyper", device=args.device)
        elif args.demo:
            profile["input"].update(type="demo")
        elif args.images:
            profile["input"].update(type="images", **{k: str(Path(p).expanduser().resolve())
                                                      for k, p in zip(("a", "b"), args.images)})
        elif args.video:
            profile["input"].update(type="video", path=str(Path(args.video).expanduser().resolve()), layout="sbs", loop=False)
        elif args.ros2:
            profile["input"].update(type="ros2", topics=args.ros2, max_skew_ms=1)
        for key in ("hfov", "pitch"):
            if getattr(args, key) is not None:
                profile["inference_view"][key] = getattr(args, key)
        if not args.no_record:
            if args.dataset is None:
                from ..common.paths import ROOT
                args.dataset = ROOT / "output" / "gesture" / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
            args.dataset = args.dataset.expanduser().resolve()
            profile["input"]["preserve_raw"] = not args.no_raw
            if not args.no_viz and args.save is None:
                args.save = args.dataset / "viewer.rrd"
            if args.metrics is None:
                args.metrics = args.dataset / "metrics.json"
            from .preflight import preflight, format_report
            report = preflight(profile, args.dataset, min_free_bytes=int(args.min_free_gb * 1024**3),
                               seconds=args.seconds, require_viewer=not args.no_viz,
                               require_camera=not args.preflight)
            print(format_report(report), flush=True)
            if not report["ok"]:
                return 1
            if args.preflight:
                return 0
        from .app import run
        run(profile, args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
