"""Independent camera workers with one preview, session and shutdown owner."""
from contextlib import ExitStack
import functools
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace

import cv2

from .storage import write_json, new_session, status, finish_replay
from .mode_viewer import BatchRecording, GestureSink, AprilGridSink, UnifiedViewer
from .preview_transport import PreviewRing, unpack_images


class WorkerStop:
    """A worker finishing its own loop must not stop the other cameras."""
    def __init__(self, parent):
        self.parent, self.local = parent, threading.Event()

    def is_set(self):
        return self.local.is_set() or self.parent.is_set()

    def set(self):
        self.local.set()

    def wait(self, timeout=None):
        end = None if timeout is None else time.monotonic()+timeout
        while not self.is_set():
            if end is not None and time.monotonic() >= end:
                return False
            self.local.wait(.02 if end is None else min(.02, max(0., end-time.monotonic())))
        return True


def worker_arguments(args, root):
    """Keep standalone defaults explicit; only the parent creates a viewer/RRD."""
    return SimpleNamespace(**{
        **vars(args), 'no_record': not args.record, 'no_viz': args.no_viz and not (args.record and args.mode == 'aprilgrid'),
        'headless': True, 'save': None, 'connect': None, 'web': False,
        'session': str(root/'data'), 'dataset': root/'data' if args.record and not getattr(args,'manual_record',False) else None,
        'dataset_fps': None, 'episode_frames': 900, 'min_free_gb': 2.,
        'task': getattr(args, 'task', 'Multi-camera gesture capture'),
        'complete_inference': False, 'metrics': root/'metrics.json',
        'stereo_only': False, 'rebuild': False,
    })


def worker(role, mode, data, args, root, preview, stop, origin, imu_queue=None):
    """Runs in a spawn process, owning all camera/model/native resources."""
    root = Path(root)
    os.setsid()  # Terminal Ctrl+C is handled once by the supervisor.
    with (root/'worker.log').open('w', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        cv2.setNumThreads(1)
        result = dict(role=role, serial=data['native']['serial'], status='starting')
        bridge = None
        preview_transport = None
        try:
            if stop.is_set():
                result['status'] = 'cancelled'
                return
            options = worker_arguments(args, root)
            if getattr(args,'manual_record',False):
                options.no_record = True
                options.recording_control = str(Path(args.recording_control_root)/role)
            local_stop = WorkerStop(stop)
            profile = data['profile']
            # ISP is owned by the parent's live viewer. Worker batches and
            # durable recording always contain the original algorithm RGB.
            profile['input']['preserve_display_raw'] = bool(data.get('display_raw', False))
            if mode == 'aprilgrid':
                profile['viewer']['display_isp']['enabled'] = False
            eyes = [ch['camera'] for ch in profile.get('channels', [])]
            control = None
            if mode == 'aprilgrid' and getattr(args,'manual_record',False):
                from .manual import WorkerControl
                control = WorkerControl(options.recording_control)
            if data.get('preview_transport') is not None:
                preview_transport = PreviewRing(data['preview_transport'])
            bridge = BatchRecording(preview, role, origin, demo=args.demo, eye_names=eyes,
                                    reliable=args.record and mode == 'aprilgrid', control=control,
                                    transport=preview_transport, imu_queue=imu_queue, attitude_alive=data.get('attitude_alive'))
            if control:
                control.finish = lambda: bridge.recording_end(control.token)
            if mode == 'gesture':
                from ..gesture.worker_environment import prepare_worker_environment, BackendReporter
                prepare_worker_environment(profile['gesture'])
                options.inference_reporter = BackendReporter(root/'inference-backend.json')
                from ..gesture.app import run
                options.cameras = data['cameras']
                if args.demo:
                    profile['input'].update(type='demo')
                    options.frames = args.frames or 12
                profile['input']['preserve_raw'] = bool(args.record and not getattr(args, 'no_raw', False))
                metrics = run(profile, options, stop_event=local_stop,
                              viewer_factory=functools.partial(GestureSink, recording=bridge))
                if metrics['viewer_errors']:
                    raise RuntimeError(f'Preview worker failed: {metrics["viewer_errors"]}')
                result['summary'] = metrics
            elif mode == 'aprilgrid':
                from ..aprilgrid.app import run
                from ..aprilgrid.config import Board
                board = Board(data['board_parameters'])
                if args.demo:
                    options.frames = args.frames or 30
                profile['unified_preview'] = True
                result['summary'] = run(profile, data['cameras'], board, options, stop_event=local_stop,
                    viewer_factory=functools.partial(AprilGridSink, recording=bridge))
            else:
                raise ValueError(f'Unsupported multi-camera worker mode: {mode}')
            result['status'] = 'stopped' if stop.is_set() else 'completed'
        except BaseException as exc:
            if stop.is_set() and isinstance(exc, RuntimeError) and str(exc) in (
                    'No input frames received', 'No stereo frames were processed'):
                result['status'] = 'cancelled'
            else:
                result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
        finally:
            result['preview'] = dict(sent=bridge.sent, dropped=bridge.dropped) if bridge else {}
            if bridge and bridge.imu_stats:
                result['imu_preview'] = dict(bridge.imu_stats)
            write_json(root/'result.json', result)
            # Parent keeps draining until every child exits, including shutdown.
            preview.close()
            preview.join_thread()
            if imu_queue is not None:
                if data.get('attitude_alive') is not None and not data['attitude_alive'].is_set():
                    imu_queue.cancel_join_thread()
                imu_queue.close()
                imu_queue.join_thread()
            if preview_transport is not None:
                preview_transport.close()


def force_stop(process, root):
    """Fallback after orderly shutdown times out; mark the session incomplete."""
    try:
        if os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGKILL)  # Includes this worker's encoders.
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.join(timeout=2)


def run(cfg, devices, args):
    if args.mode == 'reconstruction':
        return run_reconstruction(cfg, devices, args)
    from .transport import check_devices
    if not args.demo:
        check_devices(devices)
    manual_mode = getattr(args,'manual_record',False)
    immediate = args.record and not manual_mode
    with ExitStack() as resources:
        if immediate:
            root = new_session(args.mode, args.session)
            print(f'[session] {root}', flush=True)
        else:
            root = Path(resources.enter_context(tempfile.TemporaryDirectory(prefix='ego2-'+args.mode+'-')))
            print('[preview] 尚未开始录制；退出后清理预览临时缓存。', flush=True)
        session_root = root
        preview_root = root
        if immediate:
            root = session_root/'_work'
            root.mkdir()
        metadata = dict(version=2, status='recording', mode=args.mode, finished=False, synthetic_only=args.demo,
            devices={r:d['native']['serial'] for r,d in devices.items()}, config=cfg,
            clock='common host-monotonic receipt timeline; original source/device times retained per camera; not hardware synchronization',
            preview='display is independent of durable capture; dataset matching retains source times and validity masks')
        write_json(session_root/'session.json', {**(json.loads((session_root/'session.json').read_text()) if (session_root/'session.json').exists() else {}), **metadata})
        if immediate: status(session_root,'recording')
        for role in devices:
            (root/role).mkdir()
        if args.mode == 'gesture' and immediate:
            from ..gesture.preflight import preflight, format_report
            for role, data in devices.items():
                profile = data['profile']
                if args.demo: profile['input']['type'] = 'demo'
                profile['input']['preserve_raw'] = not getattr(args, 'no_raw', False)
                report = preflight(profile, root/role/'data', require_camera=not args.demo,
                                   require_viewer=not args.no_viz, seconds=args.seconds)
                print(f'[{role}] '+format_report(report), flush=True)
                if not report['ok']:
                    metadata.update(status='incomplete',error=f'{role}: gesture preflight failed')
                    write_json(session_root/'session.json', {**(json.loads((session_root/'session.json').read_text()) if (session_root/'session.json').exists() else {}), **metadata})
                    raise RuntimeError(metadata['error'])
        viewer = None
        save_replay = immediate and args.mode == 'aprilgrid'
        if save_replay or (not args.no_viz and not args.headless):
            viewer_args = SimpleNamespace(**{**vars(args), 'headless': args.headless or args.no_viz})
            try:
                viewer = UnifiedViewer(cfg, viewer_args, session_root/'replay.rrd' if save_replay else None)
            except BaseException as exc:
                if args.record:
                    status(session_root, 'incomplete', error=f'Viewer setup: {exc}')
                raise
            resources.callback(viewer.close)
        context = mp.get_context('spawn')
        stop = context.Event()
        origin = time.monotonic()
        metadata['clock_origin_monotonic'] = origin
        if immediate: status(session_root,'recording',clock_origin_monotonic=origin)
        controller = file_viewer = None
        save_work = root
        dataset_origin = origin
        stopped_batches = set()
        if manual_mode:
            from .manual import ManualController, WorkerGroup
            group = WorkerGroup(root/'_control',devices)
            args = SimpleNamespace(**{**vars(args),'recording_control_root':str(root/'_control')})
            def prepare(target, token):
                nonlocal file_viewer
                write_json(target/'session.json',{**json.loads((target/'session.json').read_text()),**metadata,'status':'preparing'})
                if args.mode == 'aprilgrid':
                    file_options = SimpleNamespace(**{**vars(args),'headless':True,'connect':None,'web':False})
                    file_viewer = UnifiedViewer(cfg,file_options,target/'replay.rrd')
                stopped_batches.clear()
                group.prepare(target,token)
            def commit(start):
                nonlocal session_root,save_work,dataset_origin
                session_root = controller.root
                save_work = session_root/'_work'
                dataset_origin = start
                metadata.update(clock_origin_monotonic=start,preview_origin_monotonic=origin,manual_recording=True)
                group.commit(start)
            def abort():
                nonlocal file_viewer,session_root,save_work
                group.abort()
                if file_viewer: file_viewer.close(); file_viewer = None
                session_root,save_work = preview_root,root
            def ready():
                stopped=[r for r,p in processes.items() if not p.is_alive()]
                if stopped: raise RuntimeError('相机采集已结束，无法开始录制：'+', '.join(stopped))
                return group.ready()
            def stop_recording(end):
                group.stop(end)
            def stopped():
                if not group.stopped():
                    if any(not p.is_alive() for p in processes.values()):
                        raise RuntimeError('相机采集在确认停止录制前退出，缓存已保留。')
                    return False
                return args.mode != 'aprilgrid' or stopped_batches == set(devices)
            def finalize():
                nonlocal file_viewer,session_root,save_work
                target,work,record_origin = session_root,save_work,dataset_origin
                saved_viewer,file_viewer = file_viewer,None
                session_root,save_work = preview_root,root
                for role in devices:
                    source=root/role/'worker.log'
                    if source.exists():
                        import shutil
                        shutil.copy2(source,work/role/'worker.log')
                status(target,'finalizing',record_stop_monotonic=controller.stop_time)
                def finish():
                    if saved_viewer: saved_viewer.close()
                    if args.mode == 'gesture':
                        from .dataset import finish_gesture
                        finish_gesture(target,work,cfg,record_origin,False,progress=controller.report_progress)
                    else:
                        finish_replay(target,target/'replay.rrd',work)
                return finish
            controller = ManualController(args.mode,args,prepare,ready,commit,abort,
                                          stop=stop_recording,stopped=stopped,finalize=finalize)
            resources.callback(controller.close)
        previous = {}
        def request_stop(*_):
            stop.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, request_stop)
            resources.callback(signal.signal, sig, previous[sig])
        processes, queues, results = {}, {}, {}
        backend_status = {}
        preview_transports = {}
        imu_queues = {}
        attitude = None
        imu_failed = set()
        failure = None
        stop_time = None
        report_at = origin

        if viewer:
            viewer.preview_origin = origin

        def drain_imu():
            # Small dedicated queues keep waveform traffic out of the image
            # transport and bypass image ISP/JPEG preparation entirely.
            for role, pending in imu_queues.items():
                for _ in range(2):
                    try:
                        packet = pending.get_nowait()
                    except queue.Empty:
                        break
                    if viewer and role not in imu_failed:
                        try:
                            viewer.consume_imu(packet)
                        except Exception as exc:
                            imu_failed.add(role)
                            print(f'[imu-preview] {role}: plotting disabled: {exc}', flush=True)
        def drain():
            drain_imu()
            for role, pending in queues.items():
                # Bounded fairness: one busy device cannot starve another.
                for _ in range(4):
                    try:
                        blob = pending.get_nowait()
                    except queue.Empty:
                        break
                    if role in preview_transports:
                        import pickle
                        blob = unpack_images(pickle.loads(blob), preview_transports[role])
                        if blob is None:
                            continue  # Complete stale frame + annotations dropped together.
                    if viewer:
                        batch=viewer.consume(blob)
                    else:
                        import pickle
                        batch=blob if isinstance(blob, dict) else pickle.loads(blob)
                    if controller and 'recording_stopped' in batch and batch['recording_stopped'] == group.token:
                        stopped_batches.add(role)
                    if controller and controller.active and file_viewer:
                        file_viewer.consume(blob,cutoff=controller.started_at,end=controller.stop_time)

            if viewer and hasattr(viewer, 'tick'):
                viewer.tick(time.monotonic()-origin)

        try:
            if args.mode == 'gesture' and viewer and not args.demo:
                from .orientation_service import AttitudeService
                attitude = AttitudeService(context, {r:d['profile'] for r,d in devices.items()}, origin, root/'attitude-worker.log')
                imu_queues.update(attitude.outputs)
            for role, data in devices.items():
                # OpenCV's dictionary is not pickleable; reconstruct it inside
                # the spawned worker from the validated physical board data.
                payload = {k:v for k,v in data.items() if k != 'board'}
                payload['display_raw'] = bool(viewer and viewer.display_processors.get(role))
                if args.mode == 'gesture' and viewer and not args.demo:
                    try:
                        transport = PreviewRing.create(context)
                    except OSError as exc:
                        print(f'[preview] {role}: 共享内存不可用，使用原预览通道：{exc}', flush=True)
                    else:
                        preview_transports[role] = transport
                        resources.callback(transport.close)
                        payload['preview_transport'] = transport.spec
                if 'board' in data:
                    payload['board_parameters'] = data['board'].data
                preview = context.Queue(maxsize=2)
                queues[role] = preview
                imu_queue = None
                if attitude is not None:
                    imu_queue = attitude.inputs[role]
                    payload['attitude_alive'] = attitude.alive
                process = context.Process(target=worker, name='ego2-'+role,
                    args=(role,args.mode,payload,args,str(root/role),preview,stop,origin,imu_queue))
                process.start()
                processes[role] = process
                print(f'[start] {role} {data["native"]["serial"]}: {args.mode}', flush=True)
                if imu_queue is not None:
                    print(f'[attitude] {role}: Mahony cam0 relative quaternion wxyz; hold still 2.5 s; '
                          'device-rate fusion; 10 Hz batches; heading may drift.', flush=True)
            while len(results) < len(processes):
                if controller: controller.poll()
                drain()
                if attitude is not None and attitude.ready.is_set() and not attitude.backend_reported:
                    attitude.backend_reported = True
                    for role, metrics in attitude.metrics().items():
                        backend = 'native CPU batch' if metrics['native_cpu'] else 'Python CPU'
                        print(f'[attitude] {role}: {backend}; independent fusion process', flush=True)
                if attitude is not None and not attitude.healthy() and not attitude.failure_reported:
                    attitude.failure_reported = True
                    print('[attitude] worker exited; camera/recording continue, attitude unavailable', flush=True)
                    if viewer and viewer.imu_logger:
                        for role in devices:
                            try:
                                viewer.imu_logger.unavailable(role, time.monotonic()-origin, 'ERROR: attitude worker stopped')
                            except Exception as exc:
                                imu_failed.add(role)
                                print(f'[imu-preview] {role}: plotting disabled: {exc}', flush=True)
                now = time.monotonic()
                if stop.is_set() and stop_time is None:
                    stop_time = now
                    print('[stop] 正在停止相机；已开始的保存会继续，完成后退出…', flush=True)
                for role, process in processes.items():
                    if role in results:
                        continue
                    if process.is_alive() and stop_time is not None and now-stop_time > 120:
                        force_stop(process,root/role)
                        failure = f'{role}: shutdown timed out; recording may be incomplete'
                    if process.is_alive():
                        continue
                    process.join()
                    result_path = root/role/'result.json'
                    result = json.loads(result_path.read_text()) if result_path.is_file() else dict(
                        status='failed', error=f'Worker exited {process.exitcode} without a result')
                    if process.exitcode:
                        result.update(status='failed', error=f'Worker exit code {process.exitcode}')
                    results[role] = result
                    if result['status'] == 'failed':
                        failure = failure or f'{role}: {result.get("error", "worker failed")}'
                        log = root/role/'worker.log'
                        print(f'[error] {role}: '+str(result.get('error'))+'\n'+
                              ('\n'.join(log.read_text(errors='replace').splitlines()[-16:]) if log.exists() else ''), flush=True)
                        stop.set()
                    print(f'[done] {role}: {result["status"]}', flush=True)
                    if viewer:
                        # Drain before marking stopped, so a final frame cannot
                        # overwrite the worker's final status.
                        drain()
                        viewer.state(role, now-origin, result['status'].upper())
                if now-report_at > 5:
                    if args.mode == 'aprilgrid':
                        for role in devices:
                            try:
                                perf = json.loads((root/role/'metrics.json').read_text())
                            except (OSError, ValueError):
                                continue
                            print(f'[aprilgrid-perf] {role}: {perf["status"]} | '
                                  f'decode={perf["decode_ms"]:.0f} flow={perf["flow_ms"]:.0f} '
                                  f'solve={perf["solve_ms"]:.0f} overlays={perf["overlays_ms"]:.0f} '
                                  f'preview-prep={perf["preview_ms"]:.0f} ms', flush=True)
                    if args.mode == 'gesture':
                        from ..gesture.worker_environment import display_backend_status
                        for role in devices:
                            backend_status[role] = display_backend_status(
                                root/role/'inference-backend.json', role, backend_status.get(role))
                    raw_logger = getattr(viewer, 'raw_imu_logger', None)
                    if raw_logger is not None:
                        try:
                            (root/'imu-preview-metrics.json').write_text(json.dumps(
                                dict(elapsed_s=now-origin, roles=raw_logger.metrics()), indent=2, allow_nan=False))
                        except OSError as exc:
                            print(f'[imu-preview] metrics unavailable: {exc}', flush=True)
                    body = getattr(viewer, 'body_preview', None)
                    if body is not None:
                        metrics = {**body.stats, **body.summary()}
                        try:
                            (root/'body-preview-metrics.json').write_text(json.dumps(metrics, indent=2))
                        except OSError as exc:
                            print(f'[body-preview] metrics unavailable: {exc}', flush=True)
                        print(f'[body-preview] drawn={body.stats["rendered"]} '
                              f'accepted={body.selector.stats["accepted"]} '
                              f'rejected={body.selector.stats["rejected"]} '
                              f'switches={body.selector.stats["switches"]} '
                              f'pose-wait={body.stats["pose_wait"]} '
                              f'clock-rejected={body.stats["clock_rejected"]} '
                              f'filters={dict(body.selector.reasons)}', flush=True)
                    print('[capture] '+', '.join(f'{r}: {results[r]["status"] if r in results else "running"}'
                                                for r in devices), flush=True)
                    size = sum(p.stat().st_size for p in save_work.rglob('*') if p.is_file())
                    if size > cfg['recording']['max_gib']*1024**3:
                        failure = 'Session size limit reached; stopping and finalizing data'
                        stop.set()
                    report_at = now
                time.sleep(.02)
            drain()
        except BaseException as exc:
            failure = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            stop.set()
            deadline = time.monotonic()+120
            # Even if the viewer failed, consume pending queues so children can
            # flush their feeder threads and finish recording.
            while any(p.is_alive() for p in processes.values()) and time.monotonic() < deadline:
                if attitude is not None: attitude.healthy()
                for pending in (*queues.values(), *imu_queues.values()):
                    try:
                        while True: pending.get_nowait()
                    except queue.Empty:
                        pass
                time.sleep(.05)
            for role, process in processes.items():
                if process.is_alive():
                    force_stop(process,root/role)
                    failure = failure or f'{role}: shutdown timed out'
                process.join(timeout=2)
            if attitude is not None:
                attitude.close(drain_imu)
                metadata['attitude_service'] = attitude.metrics()
            for pending in (*queues.values(), *imu_queues.values()):
                pending.close()
                pending.join_thread()
            for role in devices:
                path = root/role/'result.json'
                if role not in results and path.is_file():
                    results[role] = json.loads(path.read_text())
            if controller: controller.wait_finalization()
            metadata.update(finished=False, status='incomplete' if failure else 'finalizing', error=failure, results=results)
            write_json(session_root/'session.json', {**(json.loads((session_root/'session.json').read_text()) if (session_root/'session.json').exists() else {}), **metadata})
        if failure:
            raise RuntimeError(failure)
        if immediate or (controller and controller.active):
            try:
                if viewer: viewer.close()
                if file_viewer: file_viewer.close()
                if manual_mode:
                    for role in devices:
                        for name in ('worker.log','result.json','metrics.json'):
                            source=root/role/name
                            if source.exists():
                                import shutil
                                shutil.copy2(source,save_work/role/name)
                status(session_root, 'finalizing')
                if args.mode == 'gesture':
                    from .dataset import finish_gesture
                    if controller: controller.save_started = time.monotonic()
                    finish_gesture(session_root, save_work, cfg, dataset_origin, args.demo and not manual_mode,
                                   progress=controller.report_progress if controller else None)
                else:
                    finish_replay(session_root, session_root/'replay.rrd', save_work)
            except BaseException as exc:
                status(session_root, 'incomplete', error=str(exc))
                raise
            print(f'[saved] {session_root}', flush=True)
            return json.loads((session_root/'session.json').read_text())
        return metadata


def run_reconstruction(cfg, devices, args):
    """Single-camera mode uses the original reconstruction app and viewer."""
    from ..reconstruction.app import run as reconstruct
    from .transport import check_devices
    if len(devices) != 1:
        raise ValueError('三维重建仅支持一台相机')
    if not args.demo:
        check_devices(devices)
    role, data = next(iter(devices.items()))
    if getattr(args,'manual_record',False):
        from .reconstruction_recording import ReconstructionRecording
        with tempfile.TemporaryDirectory(prefix='ego2-reconstruction-preview-') as temporary:
            runtime=ReconstructionRecording(cfg,data['profile'],args)
            options=SimpleNamespace(**{**vars(args),'session':str(Path(temporary)/'data'),
                'stereo_only':False,'frames':args.frames or 12,'recording_runtime':runtime})
            error=None
            try:
                reconstruct(data['profile'],options)
            except BaseException as exc:
                error=exc
                raise
            finally:
                result=runtime.finish(error)
            return result
    with ExitStack() as resources:
        if args.record:
            root = new_session('reconstruction', args.session)
        else:
            root = Path(resources.enter_context(tempfile.TemporaryDirectory(prefix='ego2-reconstruction-')))
            print('[preview] 未开启保存；退出后清理重建临时缓存。', flush=True)
        session_root = root
        if args.record:
            root = session_root/'_work'
            root.mkdir()
        (root/role).mkdir()
        options = SimpleNamespace(**{**vars(args), 'session':str(root/role/'data'),
                                     'stereo_only':False, 'frames':args.frames or 12,
            'no_viz':False if args.record else args.no_viz,
            'headless':args.headless or (args.no_viz and args.record)})
        metadata = dict(version=2, status='recording', mode='reconstruction', finished=False,
                        synthetic_only=args.demo, devices={role:data['native']['serial']},
                        config=cfg)
        write_json(session_root/'session.json',{**(json.loads((session_root/'session.json').read_text()) if (session_root/'session.json').exists() else {}),**metadata})
        if args.record: status(session_root,'recording')
        error = None
        try:
            reconstruct(data['profile'], options)
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            metadata.update(finished=False, status='incomplete' if error else 'finalizing', error=error)
            write_json(session_root/'session.json',{**(json.loads((session_root/'session.json').read_text()) if (session_root/'session.json').exists() else {}),**metadata})
        if args.record:
            try:
                finish_replay(session_root, root/role/'data/viewer.rrd', root)
                print(f'[saved] {session_root}', flush=True)
            except BaseException as exc:
                status(session_root, 'incomplete', error=str(exc))
                raise
            return json.loads((session_root/'session.json').read_text())
        return metadata
