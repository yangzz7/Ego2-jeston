"""Optional live-display ISP. Algorithm and archive buffers never enter a writer.

One worker per camera, one active and one pending complete frame. Each job owns
its RAW bytes and its overlay snapshot; replacing a pending job drops both.
"""
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

from .paths import ROOT


def live_display(args):
    return not (getattr(args, 'demo', False) or getattr(args, 'no_viz', False)) and (
        not getattr(args, 'headless', False) or bool(getattr(args, 'connect', None)))


def settings_for(cfg, serial):
    settings = dict(cfg.get('viewer', {}).get('display_isp', {}))
    if not isinstance(settings.get('enabled', False), bool):
        raise ValueError('viewer.display_isp.enabled must be a boolean')
    if not settings.get('enabled', False):
        return None
    fps = settings.get('max_fps', 15)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('viewer.display_isp.max_fps must be positive and finite')
    base = Path(cfg.get('config_path', ROOT/'configs/multi_vio.yaml')).parent
    for key in ('config', 'library'):
        value = settings.get(key, 'auto')
        if value == 'auto':
            value = (ROOT/'cyperstereo_isp_portable/configs'/f'{serial}.yaml' if key == 'config'
                     else ROOT/'build/cyperstereo_isp/libcyperstereo_isp_c.so')
        path = Path(value).expanduser()
        settings[key] = str((base/path).resolve() if not path.is_absolute() else path)
    settings['max_fps'] = float(fps)
    backend = os.environ.get('EGO2_DISPLAY_BACKEND', settings.get('backend', 'auto'))
    if backend not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Display backend must be auto, cpu or cuda')
    settings['backend'] = backend
    return settings


def open_display_isp(cfg, serial):
    """Open the shared enhancement processor without starting a queue/thread."""
    settings = settings_for(cfg, serial)
    if settings is None:
        return None
    try:
        from cyperstereo_isp_portable.python.cyperstereo_isp import Isp
        # The display library is versioned separately from the original CPU
        # library used by recording/replay. An unavailable GPU falls back once.
        display_library = ROOT/'build/cyperstereo_isp_display/libcyperstereo_isp_c.so'
        use_display = display_library.is_file() and cfg.get('viewer', {}).get('display_isp', {}).get('library', 'auto') == 'auto'
        processor = None
        if use_display and settings['backend'] != 'cpu':
            try:
                processor = Isp(config=settings['config'], library=display_library, backend='cuda')
            except Exception as exc:
                print(f'[display-isp] {serial}: CUDA 不可用，回退 CPU：{exc}', flush=True)
        if processor is None:
            if use_display:
                try:
                    processor = Isp(config=settings['config'], library=display_library, backend='cpu')
                except Exception:
                    # Missing CUDA runtime/link dependencies must not disable
                    # the independently installed, original CPU enhancement.
                    processor = Isp(config=settings['config'], library=settings['library'])
            else:
                processor = Isp(config=settings['config'], library=settings['library'])
    except Exception as exc:
        print(f'[display-isp] {serial}: 使用原有画面，ISP 未启用：{exc}', flush=True)
        return None
    return processor


def create_display_isp(cfg, serial, args):
    if not live_display(args):
        return None
    processor = open_display_isp(cfg, serial)
    if processor is None:
        return None
    settings = settings_for(cfg, serial)
    print(f'[display-isp] {serial}: {processor.backend.upper()} 仅增强实时画面，最多 {settings["max_fps"]:g} FPS；算法和录制不变。', flush=True)
    return DisplayIsp(processor, max_fps=settings['max_fps'], name=serial)


class DisplayIsp:
    def __init__(self, processor, *, max_fps=15, name='camera'):
        self.processor, self.name = processor, name
        self.interval = 1/max_fps
        self.condition = threading.Condition()
        self.pending = None
        self.closing = False
        self.warned = set()
        self.submitted = self.dropped = self.rendered = self.fallback = 0
        self.skipped = self.enhanced = 0
        self.next_skip_warning = 0.
        self.thread = threading.Thread(target=self._run, name='display-isp-'+name, daemon=True)
        self.thread.start()

    def submit(self, raw, images, render, *, order=(0, 1), owned=False, require_enhanced=False):
        # A callback must close over immutable/snapshotted overlays, not a live
        # tracker. RAW is copied even when the producer lends a reusable array.
        # owned=True is reserved for private snapshots read by the supervisor.
        item = (None if raw is None else raw if owned else np.array(raw, copy=True),
                images, render, order, require_enhanced)
        with self.condition:
            if self.closing:
                return
            self.submitted += 1
            self.dropped += self.pending is not None
            self.pending = item
            self.condition.notify_all()

    def _warn(self, exc):
        reason = f'{type(exc).__name__}: {exc}'
        if reason not in self.warned:
            self.warned.add(reason)
            print(f'[display-isp] {self.name}: {reason}', flush=True)

    def _run(self):
        next_frame = 0.
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.closing)
                    if self.pending is None:
                        return
                    delay = next_frame-time.monotonic()
                    if delay > 0 and not self.closing:
                        self.condition.wait(timeout=delay)
                        continue
                    raw, originals, render, order, require_enhanced = self.pending
                    self.pending = None
                next_frame = time.monotonic()+self.interval
                images = originals
                try:
                    if raw is None:
                        raise ValueError('当前帧缺少 RAW')
                    processed = self.processor.process_transport(raw)
                    enhanced = tuple(processed[i] for i in order)
                    shapes = ([b.shape for b in originals] if originals is not None else
                              [(1024,1280,3)]*2)
                    if len(enhanced) != len(shapes) or any(
                        a.shape != shape or a.dtype != np.uint8 for a, shape in zip(enhanced, shapes)):
                        raise ValueError('ISP changed image geometry/format')
                    images = enhanced
                    self.enhanced += 1
                except Exception as exc:
                    if require_enhanced:
                        self.skipped += 1
                        if time.monotonic() >= self.next_skip_warning:
                            print(f'[display-isp] {self.name}: 预览增强暂不可用（{exc}）；'
                                  f'跳过本帧，保留上一帧及原时间戳；VIO/录制继续。累计跳过 {self.skipped} 帧。', flush=True)
                            self.next_skip_warning = time.monotonic()+5.
                        continue
                    self.fallback += 1
                    self._warn(exc)
                try:
                    render(images)
                    self.rendered += 1
                except Exception as exc:
                    # A preview sink failure must not terminate capture/recording.
                    self._warn(exc)
        finally:
            self.processor.close()

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('Display ISP did not stop; refusing concurrent sink teardown')
