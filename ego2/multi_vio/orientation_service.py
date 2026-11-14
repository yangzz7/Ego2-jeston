"""One optional attitude process; capture only forwards bounded IMU batches."""
import json
import math
import queue
import time
from pathlib import Path

import numpy as np

from .raw_imu import RawImuPublisher
from .imu_diagnostics import CaptureDiagnostics

FIELDS = ('samples', 'valid_quaternions', 'reference_segments', 'attitude_resets',
          'sparse_imu_gaps', 'dropped', 'failures', 'stale_batches', 'input_dropped',
          'processing_ms_total', 'processing_ms_max', 'queue_wait_ms_max', 'native_cpu')


class AttitudeInput(RawImuPublisher):
    """No fusion or blocking calls in the device reader. Owned small arrays only."""
    def __init__(self, pending, role, origin, alive):
        super().__init__(pending, role, origin, interval=1e-9, capacity=64)
        self.alive = alive
        self.stats.update(submit_ms_total=0., submit_ms_max=0., queue_full=0)
        self.diagnostics = CaptureDiagnostics()

    def _capture(self, pair):
        self.diagnostics.capture(pair)
        super()._capture(pair)

    def _append(self, row):
        self.diagnostics.pre_queue.set_epoch(self.stats['resets'])
        self.diagnostics.pre_queue.row(row)
        super()._append(row)

    def capture(self, pair):
        if not self.alive.is_set():
            if not self.disabled:
                self.disabled = True
                self.stats['failures'] += 1
                print(f'[attitude] {self.role}: unavailable; camera/recording continue', flush=True)
            return
        begin = time.perf_counter()
        super().capture(pair)
        elapsed = (time.perf_counter()-begin)*1000
        self.stats['submit_ms_total'] += elapsed
        self.stats['submit_ms_max'] = max(self.stats['submit_ms_max'], elapsed)

    def _packet(self, now):
        packet = super()._packet(now)
        packet['input_dropped'] = self.stats['dropped']
        packet.update(clock_offset=self.clock_offset, clock_epoch=self.stats['resets'], device_image=self.last_image)
        packet['raw_source_diagnostics'] = self.diagnostics.snapshot(self)
        return packet

    def flush(self, received=None):
        before = self.stats['batches']
        super().flush(received)
        if self.rows and before == self.stats['batches']:
            self.stats['queue_full'] += 1


class RoleProcessor:
    """Pure processor also used by ordering/overload regression tests."""
    def __init__(self, output, role, origin, profile):
        from .dashboard_transport import DashboardPublisher
        self.publisher = DashboardPublisher(output, role, origin, profile)
        self.last_dropped = 0
        self.last_report = -math.inf
        self.stats = self.publisher.stats
        self.stats.update(stale_batches=0, input_dropped=0, processing_ms_total=0.,
                          processing_ms_max=0., queue_wait_ms_max=0.,
                          native_cpu=int(getattr(self.publisher, 'native', False)))

    def consume(self, packet, now):
        pub = self.publisher
        rows = np.asarray(packet['imu'], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 8 or not len(rows):
            raise ValueError('Invalid raw attitude batch')
        begin = time.perf_counter()
        age = max(0., now-float(packet['received']))
        self.stats['queue_wait_ms_max'] = max(self.stats['queue_wait_ms_max'], age*1000)
        dropped = int(packet.get('input_dropped', 0))
        pub.append_raw(rows, packet)
        if age > .25:
            self.stats['stale_batches'] += 1
            pub._append((float(rows[-1,0]), math.nan, *([math.nan]*6)))
            if pub.estimator is not None:
                pub.estimator.reset('WAITING: attitude backlog; hold still to rebuild reference', count=False)
        else:
            if dropped > self.last_dropped:
                pub._append((float(rows[0,0]), math.nan, *([math.nan]*6)))
            pub.append_rows(rows)
            self.stats['samples'] += len(rows)
        self.last_dropped = dropped
        self.stats['input_dropped'] = dropped
        if float(packet['received'])-pub.last_flush >= pub.interval:
            pub.flush(packet['received'])
        elapsed = (time.perf_counter()-begin)*1000
        self.stats['processing_ms_total'] += elapsed
        self.stats['processing_ms_max'] = max(self.stats['processing_ms_max'], elapsed)
        if now-self.last_report >= 5.:
            self.last_report = now
            print('[imu-continuity] '+json.dumps(dict(role=pub.role, elapsed_s=now-pub.origin,
                  **pub.diagnostics()), allow_nan=False), flush=True)


def _serve(inputs, outputs, profiles, origin, shared, alive, ready, stop, log_path):
    import os
    with open(log_path, 'w', buffering=1) as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        try:
            processors = {r:RoleProcessor(outputs[r], r, origin, p) for r,p in profiles.items()}
            for role, processor in processors.items():
                shared[role][:] = [processor.stats.get(k, 0.) for k in FIELDS]
            ready.set()
            while True:
                busy = False
                # One batch per camera per pass: a busy camera cannot starve peers.
                for role, pending in inputs.items():
                    try: packet = pending.get_nowait()
                    except queue.Empty: continue
                    busy = True
                    processor = processors[role]
                    processor.consume(packet, time.monotonic())
                    shared[role][:] = [processor.stats.get(k,0.) for k in FIELDS]
                if not busy:
                    if stop.is_set(): break
                    stop.wait(.002)
            # Supervisor drains output while joining us.
            deadline = time.monotonic()+2
            while time.monotonic() < deadline:
                remaining = False
                for role, processor in processors.items():
                    processor.publisher.flush()
                    remaining |= bool(processor.publisher.rows)
                    shared[role][:] = [processor.stats.get(k,0.) for k in FIELDS]
                if not remaining: break
                time.sleep(.005)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            alive.clear()
            for pending in outputs.values():
                pending.close(); pending.join_thread()


class AttitudeService:
    def __init__(self, context, profiles, origin, log_path):
        self.inputs = {r:context.Queue(maxsize=8) for r in profiles}
        self.outputs = {r:context.Queue(maxsize=2) for r in profiles}
        self.shared = {r:context.RawArray('d', len(FIELDS)) for r in profiles}
        self.alive, self.ready, self.stop = context.Event(), context.Event(), context.Event()
        self.alive.set()
        self.process = context.Process(target=_serve, name='ego2-attitude', args=(
            self.inputs, self.outputs, profiles, origin, self.shared, self.alive, self.ready, self.stop, str(log_path)))
        self.process.start()
        self.closed = False
        self.failure_reported = False
        self.backend_reported = False

    def healthy(self):
        if not self.process.is_alive():
            self.alive.clear()
        return self.alive.is_set()

    def metrics(self):
        return {r:dict(zip(FIELDS, list(values))) for r,values in self.shared.items()}

    def close(self, drain):
        if self.closed: return
        self.closed = True
        self.stop.set()
        deadline = time.monotonic()+5
        while self.process.is_alive() and time.monotonic() < deadline:
            drain(); self.process.join(timeout=.01)
        if self.process.is_alive():
            self.process.terminate(); self.process.join(timeout=2)
            print('[attitude] shutdown timeout; derived preview incomplete', flush=True)
        drain()
        self.alive.clear()
        for pending in self.inputs.values():
            pending.cancel_join_thread(); pending.close()
