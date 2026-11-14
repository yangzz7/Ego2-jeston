"""Bounded scalar sidecar for the existing attitude output (no new capture)."""
import math
import numpy as np
from .native_orientation import NativeQuaternionPublisher
from .raw_imu import RawImuPublisher
from .imu_diagnostics import Continuity, RAW_GAP_SECONDS


class DashboardPublisher(NativeQuaternionPublisher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw = RawImuPublisher(None, self.role, self.origin, capacity=self.capacity)
        self.clock = {}
        self.input_drop_count = 0
        self.last_raw_stamp = None
        self.last_raw_device = None
        self.raw_epoch = None
        self.break_open = False
        self.pending_loss = False
        self.input_continuity = Continuity()
        self.source_diagnostics = {}
        self.raw_stats = dict(input_loss_events=0, input_loss_breaks=0, time_gap_breaks=0,
                              epoch_breaks=0, upstream_breaks=0, inserted_breaks=0,
                              duplicate_markers_suppressed=0)

    def append_raw(self, rows, packet):
        dropped = int(packet.get('input_dropped', 0))
        if dropped > self.input_drop_count:
            self.pending_loss = True
            self.raw_stats['input_loss_events'] += 1
        # Consume the counter once per batch, not once per row in that batch.
        self.input_drop_count = dropped
        epoch = packet.get('clock_epoch', 0)
        epoch_changed = self.raw_epoch is not None and epoch != self.raw_epoch
        self.raw_epoch = epoch
        self.input_continuity.set_epoch(epoch)
        if epoch_changed:
            self.last_raw_stamp = self.last_raw_device = None
        self.source_diagnostics = packet.get('raw_source_diagnostics', self.source_diagnostics)
        for row in rows:
            stamp = float(row[0])
            self.input_continuity.row(row)
            if not all(math.isfinite(float(v)) for v in row[1:]):
                # Input overflow/reset may already carry a marker. Keep that
                # boundary, without adding another before the next valid row.
                if not self.break_open:
                    self.raw._append(tuple(row))
                    self.raw_stats['upstream_breaks'] += 1
                else:
                    self.raw_stats['duplicate_markers_suppressed'] += 1
                self.break_open = True
                self.pending_loss = False
                epoch_changed = False
                continue
            gap = (self.last_raw_device is not None and
                   float(row[1])-self.last_raw_device > RAW_GAP_SECONDS)
            if (gap or self.pending_loss or epoch_changed) and not self.break_open:
                # Extra marker only; all accepted physical rows remain exact.
                # Rerun uses nanosecond time: nextafter alone can round to the
                # same time as the valid sample and erase the separator.
                self.raw._append((max(0., stamp-1e-6), math.nan, *([math.nan]*6)))
                self.raw_stats['inserted_breaks'] += 1
                self.raw_stats['time_gap_breaks'] += int(gap)
                self.raw_stats['input_loss_breaks'] += int(self.pending_loss)
                self.raw_stats['epoch_breaks'] += int(epoch_changed)
            self.raw._append(tuple(row))
            self.last_raw_stamp = stamp
            self.last_raw_device = float(row[1])
            self.break_open = False
            self.pending_loss = False
            epoch_changed = False
        self.clock = {k: packet.get(k) for k in ('clock_offset', 'clock_epoch', 'device_image')}

    def diagnostics(self):
        return dict(source=self.source_diagnostics, after_input_queue=self.input_continuity.snapshot(),
                    separators=dict(self.raw_stats), raw_buffer_evicted_rows=self.raw.stats['dropped'])

    def _packet(self, now):
        packet = super()._packet(now)
        packet['raw_imu'] = np.asarray(self.raw.rows, dtype=np.float64).reshape(-1, 8)
        packet['clock'] = dict(self.clock)
        packet['raw_diagnostics'] = self.diagnostics()
        return packet

    def flush(self, received=None):
        batches = self.stats['batches']
        super().flush(received)
        if self.stats['batches'] != batches:
            self.raw.rows.clear()
