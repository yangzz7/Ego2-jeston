"""python3 -m ego2.aprilgrid --help"""
import argparse
from pathlib import Path
import math
import sys

from .config import DEFAULT_CONFIG, load_config
from ..common.camera import add_camera_argument, validate_camera_args, print_camera_selection


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone stereo fisheye AprilGrid IDs and per-tag axes in Rerun")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_camera_argument(parser)
    parser.add_argument("--target", type=Path, help="Physical AprilGrid YAML (paths in profile are relative to profile)")
    parser.add_argument("--size-mm", type=float, help="Measured black-square edge; scales the whole printed board")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="Synthetic moving board; no camera required")
    source.add_argument("--device", help="Override Cyper capture node; cannot be combined with --sn")
    source.add_argument("--images", nargs=2, metavar=("CAM0", "CAM1"))
    source.add_argument("--video", help="Side-by-side original-resolution cam0/cam1 video")
    parser.add_argument("--session", help="New recording directory (never overwrite)")
    parser.add_argument("--no-record", action="store_true", help="Live preview only; do not create output files")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--headless", action="store_true", help="Save RRD without opening a viewer")
    parser.add_argument("--no-viz", action="store_true", help="Only diagnostics and pose JSONL")
    parser.add_argument("--no-display-isp", action="store_true", help="Show original RGB; leave detection unchanged")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--connect", help="Existing Rerun server URI")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--grpc-port", type=int, default=9876)
    args = parser.parse_args(argv)
    validate_camera_args(parser, args)
    if args.frames < 0 or not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("frames/seconds must be nonnegative and finite")
    if args.no_record and (args.session or args.headless):
        parser.error("--no-record cannot be combined with --session or --headless")
    if (args.web and args.connect) or (args.no_viz and (args.web or args.connect)):
        parser.error("Conflicting visualization options")
    if not all(1 <= p <= 65535 for p in (args.web_port, args.grpc_port)) or args.web_port == args.grpc_port:
        parser.error("Viewer ports must be distinct and in 1..65535")
    try:
        cfg, cameras, board = load_config(args.config, args.target, sn=args.sn)
        print_camera_selection(cfg)
        if args.no_display_isp:
            cfg["viewer"]["display_isp"]["enabled"] = False
        if args.size_mm is not None:
            if not math.isfinite(args.size_mm) or args.size_mm <= 0:
                raise ValueError("size-mm must be finite and positive")
            from .config import Board
            physical = dict(board.data)
            ratio = args.size_mm/1000/board.size
            physical["tagSize"] = args.size_mm/1000
            physical["page_size_m"] = (board.page_size*ratio).tolist()
            board = Board(physical)
            cfg["target_parameters"] = physical
        if args.device:
            cfg["input"].update(type="cyper", device=args.device)
        elif args.images:
            cfg["input"].update(type="images", **{key: str(Path(p).expanduser().resolve()) for key, p in zip(("a", "b"), args.images)})
            if not args.frames and not args.seconds:
                args.frames = 1
        elif args.video:
            cfg["input"].update(type="video", path=str(Path(args.video).expanduser().resolve()), layout="sbs", loop=False)
        if args.demo and not args.frames and not args.seconds:
            args.frames = 90
        # Two eyes are processed concurrently. Avoid nested OpenCV worker pools.
        import cv2
        cv2.setNumThreads(1)
        from .app import run
        run(cfg, cameras, board, args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
