"""Switch durable gesture capture on without replacing Source or Pipeline."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..common.runtime import QueuedArchive
from ..gesture.dataset import GestureDataset
from ..gesture.deferred import RawGestureDataset
from .manual import WorkerControl


class LiveGestureRecording:
    def __init__(self, directory, profile, args, source):
        self.profile, self.args, self.source = profile, args, source
        self.dataset = self.archive = None
        self.start_time = None
        self.stop_time = None
        self.lock = threading.RLock()
        self.finisher = ThreadPoolExecutor(max_workers=1,thread_name_prefix='gesture-finish')
        self.future = None
        self.capture_error = None
        self.control = WorkerControl(directory,self.prepare,self.abort,self.stop)

    def prepare(self, root):
        cfg = self.profile.get('input', {})
        writer = RawGestureDataset if cfg.get('type') == 'cyper' and cfg.get('preserve_raw') else GestureDataset
        dataset = writer(root/'data',self.profile,self.args.cameras,
            fps=self.args.dataset_fps or self.source.fps,task=self.args.task,
            episode_frames=self.args.episode_frames,min_free_bytes=int(self.args.min_free_gb*1024**3))
        with self.lock:
            self.start_time = self.stop_time = None
            self.capture_error = None
            self.dataset = dataset
            self.archive = QueuedArchive(dataset.capture).start()

    def poll(self):
        self.control.poll()

    def accepts(self, pair):
        # Commit is tiny and atomically published before its future cutoff.
        # Read it on the capture path too, independently of inference latency.
        if self.dataset is None or self.archive is None: return False
        path = self.control.directory/'command.json'
        if path.exists():
            command = json.loads(path.read_text())
            if command['id']==self.control.token:
                if command['phase'] in ('commit','stop'):
                    self.start_time = float(command.get('start_time',self.start_time))
                if command['phase']=='stop': self.stop_time = float(command['stop_time'])
        return (self.start_time is not None and pair.received >= self.start_time and
                (self.stop_time is None or pair.received < self.stop_time))

    def capture(self, pair):
        with self.lock:
            if self.accepts(pair):
                try: self.archive.submit(pair)
                except BaseException as exc:
                    self.capture_error = exc
                    raise

    def result(self, pair, stamp, hands, observations, processing_ms):
        with self.lock:
            if self.accepts(pair): self.dataset.result(pair,stamp,hands,observations,processing_ms)

    def abort(self):
        if self.future is not None:
            future,self.future = self.future,None
            try: future.result()
            except Exception:
                if not self.control.stop_done: raise
        self.detach(error='Recording preparation cancelled')()

    def detach(self, error=None):
        with self.lock:
            archive,self.archive = self.archive,None
            dataset,self.dataset = self.dataset,None
            started = self.start_time
            submission_failure = self.capture_error
            failure = error or self.capture_error
            self.capture_error = None
            self.start_time = None
        def finish():
            if archive is not None:
                if dataset is not None:
                    dataset.extra['archive_peak_pending'] = archive.peak_pending
                    dataset.extra['archive_queue_capacity'] = archive.pending.maxsize
                try: archive.close()
                except Exception as exc:
                    if dataset: dataset.close(error=str(exc))
                    raise
            if dataset is not None:
                dataset.close(error=failure or ('Recording never started' if started is None else None))
            if submission_failure is not None:
                raise RuntimeError(f'Recording capture failed: {submission_failure}') from submission_failure
        return finish

    def stop(self):
        self.future = self.finisher.submit(self.detach())
        return self.future

    def close(self, error=None):
        try:
            if self.future is not None: self.future.result()
            self.detach(error)()
            self.control.poll()  # Publish a completed stop even as the worker exits.
        finally:
            self.finisher.shutdown(wait=True)
