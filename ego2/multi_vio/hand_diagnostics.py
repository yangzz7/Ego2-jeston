"""Bounded, asynchronous diagnostics for the derived hand model only."""
import copy
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import queue
import threading
import time
import uuid


class HandDiagnostics:
    def __init__(self, directory=None, max_bytes=4*1024*1024, backups=2):
        directory = Path(directory) if directory else Path.home()/'.local/state/ego2/hand-model'
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory/'events.jsonl'
        self.session = uuid.uuid4().hex[:12]
        self.handler = RotatingFileHandler(self.path, maxBytes=max_bytes, backupCount=backups)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        # Handler errors are counted and reported, not silently swallowed.
        self.handler.handleError = self._write_error
        self.queue = queue.Queue(maxsize=2048)
        self.stopped = threading.Event()
        self.stats = dict(queued=0, written=0, dropped=0, errors=0)
        self.thread = threading.Thread(target=self._run, name='hand-model-diagnostics', daemon=True)
        self.thread.start()
        self.emit(dict(kind='session', reason='start', stamp=0.))

    def _write_error(self, record):
        self.stats['errors'] += 1
        if self.stats['errors'] == 1:
            print('[hand-model] diagnostic write failed; camera/model continue', flush=True)

    def emit(self, event):
        if self.stopped.is_set(): return
        row = dict(copy.deepcopy(event), session=self.session, wall_time=time.time())
        try:
            self.queue.put_nowait(row)
            self.stats['queued'] += 1
        except queue.Full:
            self.stats['dropped'] += 1
            if self.stats['dropped'] == 1:
                print('[hand-model] diagnostic queue full; lost event count is reported', flush=True)

    def _run(self):
        try:
            while not self.stopped.is_set() or not self.queue.empty():
                try: row = self.queue.get(timeout=.1)
                except queue.Empty: continue
                try:
                    record = logging.LogRecord('hand-model', logging.INFO, '', 0,
                        json.dumps(row, ensure_ascii=False), (), None)
                    errors = self.stats['errors']
                    self.handler.emit(record)
                    if errors == self.stats['errors']: self.stats['written'] += 1
                except Exception:
                    self._write_error(None)
                finally:
                    self.queue.task_done()
        finally:
            self.handler.close()

    def close(self, summary=None):
        if self.stopped.is_set(): return
        self.emit(dict(kind='session', reason='end', summary=summary, diagnostics=dict(self.stats)))
        self.stopped.set()
        self.thread.join(timeout=2.)
        if self.stats['dropped'] or self.stats['errors']:
            print(f'[hand-model] diagnostic counters: {self.stats}', flush=True)
        if self.thread.is_alive():
            print('[hand-model] diagnostic flush pending; shutdown continues', flush=True)
