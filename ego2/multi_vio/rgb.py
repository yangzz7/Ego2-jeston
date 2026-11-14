"""Preview-only stereo RGB: bounded capture -> display ISP -> Rerun.

There is no recognizer, pose estimator, dataset writer or annotation stream.
Each camera owns its capture and processing threads; slow preview processing
replaces the pending pair instead of accumulating old frames.
"""
from contextlib import ExitStack
import copy
import threading
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as b

from ..common.display_isp import open_display_isp
from ..common.runtime import LatestPairs
from ..common.sources import Source
from ..common.viewer import ViewerConnection, TIMELINE
from .preview_transport import encode_image
from .roles import VIEWER_LABELS


def build_blueprint(cfg):
    rows = []
    for role in cfg['viewer']['row_order']:
        sn = cfg['devices'][role]['sn'] or 'DEMO'
        rows.append(b.Horizontal(*[
            b.Spatial2DView(name=f'{VIEWER_LABELS[role]} · {sn} | {eye} / {cam}',
                            origin=f'/cameras/{role}/{cam}', contents='$origin/rgb')
            for eye, cam in (('Left', 'cam1'), ('Right', 'cam0'))]))
    return b.Blueprint(b.Vertical(*rows, row_shares=[1]*len(rows)),
                       b.TimePanel(state='expanded', timeline=TIMELINE, play_state='following'),
                       b.BlueprintPanel(state='hidden'), b.SelectionPanel(state='hidden'),
                       auto_layout=False, auto_views=False)


class RgbViewer(ViewerConnection):
    def __init__(self, cfg, args, *, save=None):
        self.recording = rr.RecordingStream('Ego2 · 纯RGB画面')
        self.blueprint = build_blueprint(cfg)
        self.web_process = None
        self.quality = cfg['viewer']['jpeg_quality']
        self.lock = threading.Lock()
        try:
            self._configure(cfg, args.headless, save, args.connect,
                            args.web, args.web_port, args.grpc_port)
        except BaseException:
            self.close()
            raise

    def images(self, role, stamp, images, channels):
        # Encode outside the lock, then publish the stereo pair at one time.
        encoded = [encode_image(image, self.quality) for image in images]
        with self.lock:
            self.recording.reset_time()
            self.recording.set_time(TIMELINE, duration=max(0., stamp))
            for channel, image in zip(channels, encoded):
                self.recording.log(f'/cameras/{role}/{channel["camera"]}/rgb', image)


def enhanced_images(processor, pair, profile):
    if processor is None:  # Synthetic diagnostics only.
        return pair.images
    if pair.display_transport is None:
        raise RuntimeError('纯RGB画面缺少相机 RAW，无法生成增强画面')
    images = tuple(processor.process_transport(pair.display_transport))
    cfg = profile['input']
    shape = (int(cfg['height']), int(cfg['width']), 3)
    if len(images) != 2 or any(a.shape != shape or a.dtype != np.uint8 for a in images):
        raise RuntimeError('RGB ISP returned invalid stereo geometry or pixel format')
    return images


def camera_loop(role, data, cfg, args, processor, viewer, origin, stop):
    profile = copy.deepcopy(data['profile'])
    if args.demo:
        profile['input']['type'] = 'demo'
    source = Source(profile, data['cameras'], decode_images=args.demo)
    capture = LatestPairs(source, stop)
    count, next_frame = 0, 0.
    elapsed_ms, ages_ms = [], []
    interval = 1 / cfg['viewer']['display_isp']['max_fps']
    limit = args.frames or (120 if args.demo and not args.seconds else 0)
    try:
        capture.start()
        while not stop.is_set() and (not limit or count < limit):
            # Wait first, then obtain the newest complete stereo pair.
            if stop.wait(max(0., next_frame-time.monotonic())):
                break
            pair = capture.read()
            if pair is None:
                break
            start = time.monotonic()
            next_frame = start + interval
            images = enhanced_images(processor, pair, profile)
            stamp = pair.stamp-origin if not args.demo else pair.stamp
            if viewer is not None:
                viewer.images(role, stamp, images, profile['channels'])
            count += 1
            now = time.monotonic()
            elapsed_ms.append((now-start)*1000)
            ages_ms.append((now-pair.received)*1000)
            if len(elapsed_ms) > 600:
                del elapsed_ms[:300]
                del ages_ms[:300]
    finally:
        capture.close()
    if capture.error is not None:
        raise RuntimeError(f'{role}: camera capture failed: {capture.error}') from capture.error
    if not count:
        raise RuntimeError(f'{role}: 没有收到可显示的相机画面')
    return dict(frames=count, preview_dropped=capture.dropped,
                isp_backend=getattr(processor, 'backend', 'demo'),
                mean_process_ms=float(np.mean(elapsed_ms)),
                mean_preview_age_ms=float(np.mean(ages_ms)))


def run(cfg, devices, args, *, viewer_factory=RgbViewer):
    if not args.demo:
        from .transport import check_devices
        check_devices(devices)
    cv2.setNumThreads(1)
    print('[rgb] 纯RGB画面：仅显示所选相机的增强双目画面和时间轴；Ctrl+C 退出。', flush=True)
    stops = {role: threading.Event() for role in devices}
    results, errors, threads = {}, {}, {}
    def stop_all():
        for event in stops.values():
            event.set()
    with ExitStack() as resources:
        processors = {}
        for role, data in devices.items():
            processor = None if args.demo else open_display_isp(cfg, data['native']['serial'])
            if processor is None and not args.demo:
                raise RuntimeError(f'{role}: 纯RGB画面需要可用的 ISP 增强参数和运行库，请检查上方错误')
            processors[role] = processor
            if processor is not None:
                resources.callback(processor.close)
                print(f'[rgb] {role}: {processor.backend.upper()} ISP, '
                      f'最多 {cfg["viewer"]["display_isp"]["max_fps"]:g} FPS', flush=True)
        viewer = None
        if not args.no_viz:
            viewer = viewer_factory(cfg, args)
            resources.callback(viewer.close)
        origin = time.monotonic()
        def worker(role, data):
            try:
                results[role] = camera_loop(role, data, cfg, args, processors[role], viewer, origin, stops[role])
                print(f'[rgb] {role}: {results[role]}', flush=True)
            except Exception as exc:
                errors[role] = exc
                stop_all()
                print(f'[rgb] {role}: {type(exc).__name__}: {exc}', flush=True)
        try:
            for role, data in devices.items():
                thread = threading.Thread(target=worker, args=(role, data), name='rgb-'+role, daemon=True)
                threads[role] = thread
                thread.start()
            while any(t.is_alive() for t in threads.values()):
                if errors or (args.seconds and time.monotonic()-origin >= args.seconds):
                    break
                time.sleep(.05)
        except KeyboardInterrupt:
            print('\n[rgb] 正在关闭相机预览…', flush=True)
        finally:
            stop_all()
            deadline = time.monotonic()+12
            for thread in threads.values():
                thread.join(timeout=max(0., deadline-time.monotonic()))
            if any(t.is_alive() for t in threads.values()):
                # Do not close an ISP or recording stream still in use.
                resources.pop_all()
                raise RuntimeError('RGB camera worker did not stop; close this launch process before restarting')
        if errors:
            role, error = next(iter(errors.items()))
            raise RuntimeError(f'纯RGB画面 {role} 失败：{error}') from error
    return results
