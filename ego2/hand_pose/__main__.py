"""python3 -m ego2.hand_pose [run|scan|check|calibrate] --help"""
import argparse
from pathlib import Path
import math
import sys
from .config import DEFAULT_CONFIG, load_config, validate
from .bundle import load_bundles, synthetic_bundle
from ..common.camera import add_camera_argument, validate_camera_args, print_camera_selection


def main(argv=None):
    parser = argparse.ArgumentParser(description="Head-relative wrist marker planes, 6DoF poses and trajectories")
    parser.add_argument("command", nargs="?", choices=("run", "scan", "check", "calibrate"), default="run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_camera_argument(parser)
    parser.add_argument("--dictionary", help="OpenCV dictionary, e.g. DICT_APRILTAG_36h11")
    parser.add_argument("--border-bits", type=int, help="Black border cells: historical Kalibr print uses 2")
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--bundle-calibration", type=Path, help="Wrist geometry JSON; calibrate writes, run reads")
    parser.add_argument("--left-id", type=int, help="Switch left target to legacy single-tag mode")
    parser.add_argument("--right-id", type=int)
    parser.add_argument("--size-mm", type=float, help="Measured outer black square edge (excluding white margin)")
    parser.add_argument("--allow-mono", action="store_true", help="Allow unambiguous monocular poses; default requires stereo")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true")
    source.add_argument("--device")
    source.add_argument("--images", nargs=2, metavar=("CAM0", "CAM1"))
    source.add_argument("--video", help="SBS cam0/cam1 recording with original per-eye resolution")
    parser.add_argument("--session", help="New output directory; existing directories are never overwritten")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--headless", action="store_true", help="Save RRD without opening a viewer")
    parser.add_argument("--no-viz", action="store_true", help="Save poses/metadata only")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--connect")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--grpc-port", type=int, default=9876)
    args = parser.parse_args(argv)
    validate_camera_args(parser, args)
    if args.frames < 0 or not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("frames/seconds must be nonnegative and finite")
    if args.web and args.connect or args.no_viz and (args.web or args.connect):
        parser.error("Cannot combine --web/--connect or --no-viz with viewer connection options")
    if not all(1 <= p <= 65535 for p in (args.web_port, args.grpc_port)) or args.web_port == args.grpc_port:
        parser.error("Viewer ports must be distinct and in 1..65535")
    try:
        cfg, cameras = load_config(args.config, sn=args.sn)
        print_camera_selection(cfg)
        if args.hand != "both":
            cfg["targets"] = [t for t in cfg["targets"] if t["name"] == args.hand]
        if args.bundle_calibration:
            cfg["bundle_calibration"] = str(args.bundle_calibration.expanduser().resolve())
        if args.dictionary:
            cfg["dictionary"] = args.dictionary
        if args.border_bits is not None:
            cfg["border_bits"] = args.border_bits
        for target in cfg["targets"]:
            value = getattr(args, target["name"]+"_id", None)
            if value is not None:
                target.pop("ids", None)
                target.pop("T_wrist_endpoint", None)
                target.pop("T_wrist_marker", None)
                target["id"] = value
            if args.size_mm is not None:
                target["size_m"] = args.size_mm / 1000
        if args.allow_mono:
            cfg["quality"]["require_stereo"] = False
        validate(cfg)
        # CLI bindings plus measured size explicitly acknowledge the actual wrist tags.
        explicit = args.size_mm is not None and all(getattr(args, t["name"]+"_id", None) is not None for t in cfg["targets"])
        if args.command == "run" and not args.demo and not (cfg.get("markers_configured") is True or explicit):
            raise ValueError("先确认腕带参数：使用 scan 查看 ID，再用 --left-id N --right-id M --size-mm 实测边长；或在标定码配置（默认 calibration_targets/wrist_tags.yaml）填好 targets 并设置 markers_configured: true")
        if args.command == "check" and not args.frames and not args.seconds:
            args.seconds = 8
        if args.command == "calibrate":
            if args.allow_mono or not cfg["quality"]["require_stereo"]:
                raise ValueError("Wrist calibration requires stereo observations")
            if any("ids" not in t for t in cfg["targets"]):
                raise ValueError("Wrist calibration requires multi-tag targets")
            if not args.frames and not args.seconds:
                args.seconds = 45
        if args.demo:
            if args.command == "calibrate":
                raise ValueError("Use the bundle tests for synthetic calibration; demo must not overwrite real wrist geometry")
            for target in cfg["targets"]:
                if "ids" in target:
                    target["T_wrist_marker"] = synthetic_bundle(target)
        elif args.command == "run" and any("ids" in t for t in cfg["targets"]):
            load_bundles(cfg)
        if args.device:
            cfg["input"].update(type="cyper", device=args.device)
        elif args.images:
            cfg["input"].update(type="images", **{k: str(Path(p).expanduser().resolve()) for k, p in zip(("a", "b"), args.images)})
        elif args.video:
            cfg["input"].update(type="video", path=str(Path(args.video).expanduser().resolve()), layout="sbs", loop=False)
        if args.demo and not args.frames and not args.seconds:
            args.frames = 90
        from .app import run
        run(cfg, cameras, args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
