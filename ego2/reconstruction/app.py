"""Launch/own the native VIO process and consume bounded RGB previews."""
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import tempfile

import cv2
import numpy as np
import zmq

from .build import BUILD, ROOT, build_native, native_is_stale
from .calibration import write_orb_settings
from .session import DenseMap, write_json


def stop_backend(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            print("[warning] Backend did not stop within 45 s; final map may be incomplete", flush=True)


def create_session(cfg, args):
    session = (Path(args.session).expanduser().resolve() if args.session else
               ROOT/"output/reconstruction"/datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    # An existing run is never overwritten, including interrupted runs.
    session.mkdir(parents=True, exist_ok=False)
    (session/"keyframes").mkdir()
    effective = dict(cfg)
    checksums = {}
    for field in ("calibration", "imu_calibration"):
        src = Path(cfg[field])
        dst = session/(field+".yaml")
        shutil.copy2(src,dst)
        effective[field] = str(dst)
        checksums[field] = hashlib.sha256(dst.read_bytes()).hexdigest()
    shift = write_orb_settings(effective,session/"orb_settings.yaml")
    write_json(session/"session.json", {
        "version": 1, "serial": cfg["serial"], "created": datetime.now().astimezone().isoformat(),
        "config": effective, "calibration_sha256": checksums,
        "mode": "demo" if args.demo else "stereo" if args.stereo_only else "stereo-inertial",
        "timestamp": "device seconds; ORB time = image device time + cam1 Kalibr timeshift",
        "camera_time_shift_s": shift, "camera_order": ["cam1","cam0"],
        "pose": "T_world_camera maps native cam1 to its ORB map world, metres",
        "keyframe_images": "original resolution, Bayer demosaic + bounded gray-world balance, JPEG quality 95",
    })
    return session,effective,shift


def run(cfg, args):
    executable = BUILD/"bin/ego2_reconstruction_native"
    if not args.demo and (args.rebuild or native_is_stale(executable,cfg)):
        executable = build_native(cfg,args.jobs)
    session,cfg,shift = create_session(cfg,args)
    print(f"[session] {session}",flush=True)
    cv2.setNumThreads(2)
    with ExitStack() as resources:
        runtime = getattr(args,'recording_runtime',None)
        viewer = runtime
        if not args.no_viz and runtime is None:
            from .visualizer import Visualizer
            viewer = Visualizer(cfg,args,session/"viewer.rrd")
            resources.callback(viewer.close)
        dense = DenseMap(cfg,session,viewer)
        if runtime:runtime.dense=dense
        if args.demo:
            from .demo import run_demo
            return run_demo(cfg,args,session,dense,viewer)
        context = zmq.Context()
        resources.callback(context.term)
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE,b"")
        subscriber.setsockopt(zmq.RCVHWM,2)
        subscriber.setsockopt(zmq.LINGER,0)
        resources.callback(subscriber.close)
        socket_path = Path(f"/tmp/ego2-reconstruction-{os.getpid()}.ipc")
        endpoint = "ipc://"+str(socket_path)
        resources.callback(lambda: socket_path.unlink(missing_ok=True))
        subscriber.connect(endpoint)
        log = resources.enter_context((session/"backend.log").open("w"))
        command = [str(executable),cfg["vocabulary"],str(session/"orb_settings.yaml"),str(session),endpoint,
                   cfg["serial"],str(cfg["fps"]),str(shift),str(cfg["acceleration_scale"]),
                   str(cfg["max_imu_gap"]),str(args.seconds),"stereo" if args.stereo_only else "vio"]
        environment = dict(os.environ)
        environment.pop('EGO2_DISPLAY_RAW_RING',None)
        display_path = None
        display_reader = None
        if viewer and getattr(viewer,'wants_display_raw',False):
            display_root = resources.enter_context(tempfile.TemporaryDirectory(prefix='ego2-display-raw-'))
            display_path = Path(display_root)/'frames.bin'
            environment['EGO2_DISPLAY_RAW_RING'] = str(display_path)
        process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,
                                   start_new_session=True,cwd=session)
        resources.callback(stop_backend,process)
        origin, elapsed, frames, last_status = None,0.0,0,0.0
        snapshot = None
        started = time.monotonic()
        print("[VIO] 初始化时请缓慢移动、转动相机，观察有纹理的静态环境。Ctrl+C 保存并退出。",flush=True)
        try:
            while process.poll() is None:
                if runtime:runtime.poll()
                if subscriber.poll(100):
                    parts = subscriber.recv_multipart()
                    # Keep only complete multipart frames; metadata and pixels
                    # from different timestamps can never be paired together.
                    while subscriber.poll(0):
                        parts = subscriber.recv_multipart()
                    if len(parts) != 3:
                        raise ValueError("Invalid native stereo transport")
                    meta = json.loads(parts[0])
                    stamp = float(meta["timestamp"])
                    origin = stamp if origin is None else origin
                    elapsed = stamp-origin
                    images = [cv2.imdecode(np.frombuffer(p,np.uint8),cv2.IMREAD_COLOR) for p in parts[1:]]
                    if any(im is None for im in images):
                        raise ValueError("Corrupted preview image")
                    if viewer:
                        raw = None
                        if display_path is not None:
                            try:
                                if display_reader is None:
                                    from ..common.display_raw import DisplayRawReader
                                    display_reader = DisplayRawReader(display_path)
                                    resources.callback(display_reader.close)
                                raw = display_reader.read(meta['frame_id'])
                            except (OSError,ValueError):
                                pass  # Current original JPEG remains the fallback.
                            viewer.rgb(images,elapsed,raw_transport=raw)
                        else:
                            viewer.rgb(images,elapsed)
                    frames += 1
                map_path = session/"maps.json"
                if map_path.exists():
                    snapshot = json.loads(map_path.read_text())
                    if snapshot.get("mode") == "stereo-inertial" and snapshot.get("version",1) < 2:
                        raise RuntimeError("旧后端缺少 VIO 初始化完成标志，请使用 --rebuild 重新构建后启动")
                    dense.update(snapshot,elapsed,budget=1)
                now = time.monotonic()
                if now-last_status >= 5:
                    state = snapshot.get("state","starting") if snapshot else "loading vocabulary"
                    ready = len(dense.live)
                    phase = "初始化完成" if dense.ready else "等待 VIO 初始化完成，点云暂停"
                    print(f"[VIO] {phase} | state={state} | RGB={frames} | cloud keyframes={ready}",flush=True)
                    last_status=now
                if frames == 0 and now-started > 90:
                    raise TimeoutError(f"No RGB preview after 90 seconds; inspect {session/'backend.log'}")
        except KeyboardInterrupt:
            print("\n[session] 正在停止 VIO、保存最终位姿和关键帧深度…",flush=True)
        finally:
            stop_backend(process)
        map_path = session/"maps.json"
        if map_path.exists():
            snapshot = json.loads(map_path.read_text())
            dense.update(snapshot,elapsed,budget=None)
        if process.returncode != 0:
            tail = "\n".join((session/"backend.log").read_text(errors="replace").splitlines()[-15:])
            raise RuntimeError(f"Native backend exited {process.returncode}:\n{tail}\nFull log: {session/'backend.log'}")
        print(f'[{"cache" if runtime else "saved"}] {session}',flush=True)
        if not dense.live:
            print("[VIO] 本次没有可融合的关键帧。请检查光照、纹理和初始化运动；详情见 backend.log。",flush=True)
        return session
