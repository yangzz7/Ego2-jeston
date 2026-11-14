"""Shared ordered frame clock and bounded stereo capture worker."""
import math
import sys
import threading
import queue
import time


class QueuedArchive:
    """Keep video/disk stalls off the device reader, with a bounded FIFO.

    On overflow fail explicitly instead of blocking camera reads or silently
    dropping frames. Stop the camera producer before calling close().
    """
    def __init__(self, callback, max_pending=32):
        if max_pending < 1:
            raise ValueError("Archive queue capacity must be positive")
        self.callback = callback
        self.pending = queue.Queue(maxsize=max_pending)
        self.error = None
        self.peak_pending = 0
        self._sentinel = object()
        self._sentinel_sent = False
        self._closed = False
        self.thread = threading.Thread(target=self._write, name="ego2-archive-writer", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _write(self):
        try:
            while True:
                pair = self.pending.get()
                if pair is self._sentinel:
                    return
                self.callback(pair)
        except BaseException as exc:
            self.error = exc

    def submit(self, pair):
        if self._closed:
            raise RuntimeError("Archive queue is closed")
        if self.error is not None:
            raise RuntimeError(f"Archive writer failed: {self.error}") from self.error
        try:
            self.pending.put_nowait(pair)
        except queue.Full as exc:
            raise RuntimeError("Archive queue is full: disk/encoder cannot keep up with the camera; capture stopped instead of silently dropping data") from exc
        self.peak_pending = max(self.peak_pending, self.pending.qsize())

    def close(self, timeout=60):
        # Reject new submissions immediately, but permit a second close() to
        # finish draining after a previous timeout. Otherwise an unsent
        # sentinel would leave the worker blocked forever on its next get().
        self._closed = True
        deadline = time.monotonic() + timeout
        while not self._sentinel_sent and self.thread.is_alive() and self.error is None:
            try:
                self.pending.put(self._sentinel, timeout=min(0.1, max(0, deadline - time.monotonic())))
                self._sentinel_sent = True
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Archive writer did not drain; recording remains incomplete")
        self.thread.join(timeout=max(0, deadline - time.monotonic()))
        if self.thread.is_alive():
            raise RuntimeError("Archive writer did not stop; recording remains incomplete")
        if self.error is not None:
            raise RuntimeError(f"Archive writer failed: {self.error}") from self.error

class FrameClock:
    """Keep source intervals; never disguise clock resets as continuous tracking."""
    def __init__(self):
        self.origin = None
        self.previous = -math.inf
        self.previous_ms = -1

    def update(self, stamp):
        if not math.isfinite(stamp) or stamp <= self.previous:
            raise ValueError("Input timestamps repeated or moved backwards; restart the source/replay")
        if self.origin is None:
            self.origin = stamp
        elapsed = stamp - self.origin
        # MediaPipe only accepts integer milliseconds; preserve full precision
        # on the Rerun timeline and only round/advance the model's clock.
        milliseconds = max(self.previous_ms + 1, round(elapsed * 1000))
        self.previous, self.previous_ms = stamp, milliseconds
        return elapsed, milliseconds


class LatestPairs:
    """Single complete-pair slot. The capture thread exclusively owns Source."""
    def __init__(self, source, stop, on_pair=None):
        self.source, self.stop = source, stop
        # Archiving precedes the lossy latest-frame inference slot. Every pair
        # returned by Source is recorded, including pairs inference skips.
        self.on_pair = on_pair
        self.condition = threading.Condition()
        self.latest = None
        self.error = None
        self.done = False
        self.dropped = 0
        self.thread = threading.Thread(target=self._capture, name="ego2-gesture-capture", daemon=True)

    def _capture(self):
        try:
            self.source.open()
            while not self.stop.is_set():
                pair = self.source.read(self.stop)
                if pair is None:
                    break
                if self.on_pair is not None:
                    self.on_pair(pair)
                if self.stop.is_set():
                    break
                with self.condition:
                    if self.latest is not None:
                        self.dropped += 1
                    self.latest = pair
                    self.condition.notify_all()
        except Exception as exc:
            self.error = exc
        finally:
            try:
                self.source.close()
            except Exception as exc:
                self.error = self.error or exc
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def start(self):
        self.thread.start()
        return self

    def read(self, timeout=5.0):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.latest is not None or self.done or self.stop.is_set(), timeout=timeout)
            if not ready:
                raise TimeoutError("No complete stereo frame for 5 seconds; check camera/ROS topics")
            if self.error is not None:
                raise RuntimeError(f"Capture failed: {self.error}") from self.error
            pair, self.latest = self.latest, None
            return pair

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=10 if self.on_pair is not None else 2)
        if self.thread.is_alive():
            if self.on_pair is not None:
                raise RuntimeError("Capture did not stop within 10 s; recording remains incomplete to avoid concurrent finalization")
            print("[warning] Capture driver did not return within 2 s; worker exits with process", file=sys.stderr)
        elif self.on_pair is not None and self.error is not None:
            raise RuntimeError(f"Capture/archive failed: {self.error}") from self.error
