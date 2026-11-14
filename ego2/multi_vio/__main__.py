"""Multi-camera ORB-SLAM3 + AprilGrid + native Rerun entry point."""
import argparse
import math
import subprocess
import sys
from .config import MULTI_CONFIG, load_multi_config
from .modes import MODES, load_mode_config


def main(argv=None):
    parser=argparse.ArgumentParser(description='1–4 cameras: RGB preview, VIO, gesture, AprilGrid or reconstruction in one Rerun Viewer')
    parser.add_argument('command',nargs='?',choices=('run','build','check','replay'),default='run')
    parser.add_argument('--config',default=str(MULTI_CONFIG))
    parser.add_argument('--mode',choices=tuple(MODES),help='Initial panel mode; with --no-setup overrides the saved mode')
    parser.add_argument('--no-setup',action='store_true',help='Skip the setup panel; use saved SN mapping, then YAML as fallback')
    record=parser.add_mutually_exclusive_group()
    record.add_argument('--record',action='store_true',help='Enable manual recording (the default): preview first; Enter in native Rerun (X11) or launch terminal opens recording settings; during recording Enter stops/saves and keeps preview running (with --headless: start immediately)')
    record.add_argument('--no-record',dest='record',action='store_false',help='Preview only; disable recording controls')
    parser.set_defaults(record=True)
    parser.add_argument('--session',help='Recording panel preset, headless destination, or existing session for replay')
    parser.add_argument('--save',help='Replay only: export an independent .rrd without overwriting')
    parser.add_argument('--seconds',type=float,default=0,help='Stop after this many wall-clock seconds; 0 is unlimited')
    parser.add_argument('--frames',type=int,default=0,help='Stop after this many frames/device; demo default is 120')
    parser.add_argument('--demo',action='store_true',help='Synthetic VIO and real board detector, no camera; not an accuracy measurement')
    parser.add_argument('--no-viz',action='store_true',help='Run without Rerun, for diagnostics')
    parser.add_argument('--headless',action='store_true',help='Start saving immediately without opening Viewer; incompatible with --no-record (replay: requires --save)')
    parser.add_argument('--web',action='store_true',help='Use the official Rerun Web Viewer')
    parser.add_argument('--web-port',type=int,default=9090)
    parser.add_argument('--grpc-port',type=int,default=9876)
    parser.add_argument('--connect',help='Existing Rerun gRPC endpoint')
    parser.add_argument('--rebuild',action='store_true')
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--task',default='Multi-camera gesture capture',help='Gesture dataset task description')
    parser.add_argument('--no-raw',action='store_true',help='Gesture recording: omit packed Bayer, keep RGB/IMU/results')
    args=parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds<0 or args.frames<0 or args.jobs<1:
        parser.error('seconds/frames must be nonnegative; jobs must be positive')
    if args.command == 'replay':
        if not args.session: parser.error('replay requires --session DATA_DIRECTORY')
        if args.headless and not args.save: parser.error('headless replay requires --save FILE.rrd')
        try:
            from .replay import replay
            replay(args)
            return 0
        except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
            print(f'[error] {exc}', file=sys.stderr)
            return 1
    if args.save: parser.error('--save is for replay; use --session for the live recording destination')
    args.manual_record = args.record and not args.headless
    if (args.session or args.headless) and not args.record:
        parser.error('--session/--headless cannot be combined with --no-record')
    if (args.web and args.connect) or (args.no_viz and (args.web or args.connect)):
        parser.error('Conflicting visualization options')
    if not all(1 <= p <= 65535 for p in (args.web_port,args.grpc_port)) or args.web_port == args.grpc_port:
        parser.error('Viewer ports must be distinct and in 1..65535')
    try:
        selection=None
        mode=args.mode or 'vio'
        if not args.demo:
            from .selection import default_selection, open_launcher, default_mode
            if args.command=='run' and not (args.no_setup or args.no_viz or args.headless):
                setup=open_launcher(args.config,mode=args.mode)
                if setup is None:
                    print('[setup] 已取消，未启动采集或 Rerun。',flush=True)
                    return 0
                if setup.get('action') == 'replay':
                    return int(setup.get('exit_code', 0))
                if 'devices' in setup:
                    mode,selection=setup['mode'],setup['devices']
                else:
                    selection=setup  # Legacy programmatic callers.
            else:
                mode=args.mode or default_mode(args.config)
                selection=default_selection(args.config,mode=mode)
            from .profiles import prepare_camera_profiles
            _,problems=prepare_camera_profiles()
            for problem in problems.values():
                print(f'[camera] {problem}',file=sys.stderr)
        args.mode=mode
        if mode == 'rgb':
            if args.session:
                raise ValueError('纯RGB画面仅预览，不使用 --session；录制请使用其他模式')
            if args.headless and not (args.no_viz or args.connect or args.web):
                raise ValueError('纯RGB画面无窗口运行请使用 --no-viz，或指定 --connect')
            args.record = args.manual_record = False
        if mode=='vio':
            cfg,devices=load_multi_config(args.config,demo=args.demo,selection=selection)
        else:
            cfg,devices=load_mode_config(args.config,mode=mode,demo=args.demo,selection=selection)
        print(f'[mode] {MODES[mode]}',flush=True)
        for role,data in devices.items():
            prefix='demo calibration' if args.demo else 'camera'
            print(f'[{prefix}] {role}: {data["native"]["serial"]}',flush=True)
        if args.command=='build':
            if mode=='vio':
                from .transport import ensure_native
                print(ensure_native(devices[cfg['reference_role']]['native'],args.jobs,True))
            elif mode=='reconstruction':
                from ..reconstruction.build import build_native
                print(build_native(next(iter(devices.values()))['profile'],args.jobs))
            else:
                print('[build] This mode does not require a native VIO backend.')
        elif args.command=='check':
            if not args.demo:
                from .transport import check_devices
                check_devices(devices)
            print('[check] Configuration'+(' and USB discovery' if not args.demo else ' (synthetic demo)')+
                  ' passed. No camera stream or VIO was started.',flush=True)
        else:
            if mode=='vio':
                from .app import run
            elif mode == 'rgb':
                from .rgb import run
            else:
                from .multi_mode import run
            run(cfg,devices,args)
    except KeyboardInterrupt:
        print('\n[stop] 已取消启动。',file=sys.stderr)
        return 130
    except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as exc:
        print(f'[error] {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
