"""Usage: python3 -m ego2.reconstruction [build|export] [options]."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from .calibration import DEFAULT_CONFIG, load_config
from ..common.camera import add_camera_argument, validate_camera_args, print_camera_selection


def main(argv=None):
    parser=argparse.ArgumentParser(description="Cyperstereo ORB-SLAM3 stereo VIO + dense reconstruction + Rerun")
    parser.add_argument("command",nargs="?",choices=("run","build","export"),default="run")
    parser.add_argument("--config",default=str(DEFAULT_CONFIG))
    add_camera_argument(parser)
    parser.add_argument("--session",help="new session directory for run; existing session for export")
    parser.add_argument("--seconds",type=float,default=0,help="live capture duration; 0 means until Ctrl+C")
    parser.add_argument("--demo",action="store_true",help="synthetic images and known poses; no camera/VIO")
    parser.add_argument("--frames",type=int,default=12,help="synthetic demo frame count")
    parser.add_argument("--stereo-only",action="store_true",help="diagnose stereo SLAM without IMU (default is VIO)")
    parser.add_argument("--headless",action="store_true",help="record RRD without opening a viewer")
    parser.add_argument("--no-viz",action="store_true",help="reconstruct/save without Rerun")
    parser.add_argument("--web",action="store_true")
    parser.add_argument("--web-port",type=int,default=9090)
    parser.add_argument("--grpc-port",type=int,default=9876)
    parser.add_argument("--connect",help="connect to an existing Rerun gRPC viewer")
    parser.add_argument("--rebuild",action="store_true")
    parser.add_argument("--jobs",type=int,default=2)
    parser.add_argument("--format",choices=("points","mesh","both"),default="both")
    parser.add_argument("--voxel",type=float,default=0.02,help="export voxel size in metres")
    parser.add_argument("--open3d-python",help="Python interpreter with Open3D (export only)")
    args=parser.parse_args(argv)
    validate_camera_args(parser, args)
    import math
    if args.jobs<1 or args.frames<1 or not math.isfinite(args.seconds) or args.seconds<0 or not math.isfinite(args.voxel) or args.voxel<=0:
        parser.error("jobs/frames/voxel must be positive; seconds must be finite and nonnegative")
    try:
        if args.command=="export":
            if not args.session:
                parser.error("export requires --session")
            session=Path(args.session).expanduser().resolve()
            snapshot=json.loads((session/"maps.json").read_text())
            if not snapshot.get("final"):
                raise ValueError("Stop acquisition cleanly before exporting the final map")
            cfg=json.loads((session/"session.json").read_text())["config"]
            # Sessions are movable: prefer the preserved calibration copies.
            cfg["calibration"]=str(session/"calibration.yaml")
            cfg["imu_calibration"]=str(session/"imu_calibration.yaml")
            from .session import DenseMap
            DenseMap(cfg,session).prepare_export(snapshot)
            executable=args.open3d_python
            if not executable:
                executable=sys.executable if importlib.util.find_spec("open3d") else "/usr/bin/python3"
            result=subprocess.run([executable,"-c","import open3d"],capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError("Open3D is unavailable; install requirements-export.txt in a Python 3.10–3.12 environment and set --open3d-python")
            subprocess.run([executable,str(Path(__file__).with_name("export.py")),"--session",str(session),
                            "--format",args.format,"--voxel",str(args.voxel)],check=True)
        else:
            cfg=load_config(args.config, sn=args.sn)
            print_camera_selection(cfg)
            if args.command=="build":
                from .build import build_native
                print(build_native(cfg,args.jobs))
            else:
                from .app import run
                run(cfg,args)
    except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as exc:
        print(f"[error] {exc}",file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
