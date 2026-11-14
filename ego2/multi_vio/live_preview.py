"""Bounded live trajectory preview, independent of AprilGrid completion.

Pixels are private ring-reader snapshots. Encoding never writes to them. Board
results may arrive before or after preview, but only attach to the exact frame
that was published; each subsequent image clears the previous annotations.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
import copy
import threading
import time

from .preview_transport import encode_image


class LatestFrameWorker:
    """Fallback when ISP is disabled: one active and one replaceable job."""
    def __init__(self, max_fps, name):
        self.interval = 1 / max_fps
        self.condition = threading.Condition()
        self.pending = None
        self.closing = False
        self.dropped = 0
        self.warned = False
        self.thread = threading.Thread(target=self._run, name='vio-preview-'+name, daemon=True)
        self.thread.start()

    def submit(self, job):
        with self.condition:
            if self.closing:
                return
            self.dropped += self.pending is not None
            self.pending = job
            self.condition.notify_all()

    def _run(self):
        next_frame = 0.
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closing)
                if self.pending is None:
                    return
                delay = next_frame-time.monotonic()
                if delay > 0 and not self.closing:
                    self.condition.wait(timeout=delay)
                    continue
                job, self.pending = self.pending, None
            next_frame = time.monotonic()+self.interval
            try:
                job()
            except Exception as exc:
                if not self.warned:
                    print(f'[vio-preview] {exc}', flush=True)
                    self.warned = True

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('Trajectory preview did not stop')


def frame_key(packet):
    return packet['map_id'], packet['frame_id'], packet['capture_time']


class TrajectoryPreview:
    HISTORY = 128
    PENDING_BOARDS = 4

    def __init__(self, roles, processors, quality, max_fps, publish, annotate):
        self.processors = processors
        self.quality = quality
        self.publish, self.annotate = publish, annotate
        # Shared by every camera: at most two JPEG encoders, not two per role.
        self.encoders = ThreadPoolExecutor(max_workers=2, thread_name_prefix='vio-jpeg')
        self.fallback = {role: LatestFrameWorker(max_fps, role)
                         for role in roles if not processors.get(role)}
        self.locks = {role: threading.Lock() for role in roles}
        self.displayed = {role: OrderedDict() for role in roles}
        self.pending = {role: OrderedDict() for role in roles}
        self.last_time = {role: float('-inf') for role in roles}
        self.frames = {role: 0 for role in roles}

    def submit(self, role, packet, images, status, raw_transport=None):
        packet = copy.deepcopy(packet)
        def render(shown):
            self._render(role, packet, shown, status)
        processor = self.processors.get(role)
        if processor:
            # Backend owns one atomic RGB/RAW snapshot. A failed enhancement
            # drops the entire preview frame, never its brightness policy.
            processor.submit(raw_transport, images, render, owned=True, require_enhanced=True)
        else:
            self.fallback[role].submit(lambda: render(images))

    def _render(self, role, packet, images, status):
        futures = [self.encoders.submit(encode_image, image, self.quality) for image in images]
        # Finish both eyes before publishing either, retaining one timestamp.
        # Wait even on encoder failure so failed jobs cannot leave queued eyes.
        wait(futures)
        encoded = [future.result() for future in futures]
        key = frame_key(packet)
        with self.locks[role]:
            self.publish(role, packet, encoded, status)
            self.frames[role] += 1
            self.displayed[role][key] = None
            while len(self.displayed[role]) > self.HISTORY:
                self.displayed[role].popitem(last=False)
            self.last_time[role] = packet['capture_time']
            board = self.pending[role].pop(key, None)
            if board is not None:
                self.annotate(role, *board)
            # Dropped preview frames can never acquire overlays later.
            self.pending[role] = OrderedDict((k, v) for k, v in self.pending[role].items()
                if k[2] > packet['capture_time'])

    def board(self, role, packet, observations, track, status, processing_ms):
        board = copy.deepcopy((packet, observations, track, status, processing_ms))
        key = frame_key(packet)
        with self.locks[role]:
            if key in self.displayed[role]:
                # Historical log at the original capture time. A later image
                # already has its own clear, so this cannot leak into that frame.
                self.annotate(role, *board)
            elif packet['capture_time'] > self.last_time[role]:
                self.pending[role][key] = board
                while len(self.pending[role]) > self.PENDING_BOARDS:
                    self.pending[role].popitem(last=False)

    def close(self):
        # ISP owns render callbacks too; all producers must stop before encoders.
        for processor in self.processors.values():
            if processor:
                processor.close()
        for worker in self.fallback.values():
            worker.close()
        self.encoders.shutdown(wait=True)
