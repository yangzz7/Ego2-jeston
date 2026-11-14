"""Disposable, bounded raw-IMU preview, independent of inference and recording.

Values are the decoder's accepted measurements in physical units, without
filtering or integration. Each device clock is anchored once to an image's
common host receipt time. Subsequent samples retain their device intervals,
without USB receipt jitter. This is not hardware clock synchronization.
"""
from collections import deque
import math
import queue
import sys

import numpy as np
import rerun as rr
import rerun.blueprint as b

from ..common.viewer import TIMELINE

COLORS = ([239, 95, 98], [89, 204, 124], [86, 155, 255])
FIELDS = ('acceleration_m_s2', 'angular_velocity_rad_s')


def charts(role, label):
    """Native Rerun plots, following a ten-second window on the shared clock."""
    return [b.TimeSeriesView(
        name=f'{label} | {title}' if label else title, origin=f'/imu/{role}/{field}', contents='$origin/**',
        axis_x=b.TimeAxis(view_range=b.TimeRange(
            b.TimeRangeBoundary.cursor_relative(seconds=-10),
            b.TimeRangeBoundary.cursor_relative(seconds=0))),
        plot_legend=b.PlotLegend(corner='RightTop', visible=True))
        for field, title in zip(FIELDS, ('Accelerometer (m/s^2)', 'Gyroscope (rad/s)'))]


class RawImuPublisher:
    """Capture-thread observer: no Rerun, image copies, inference or blocking I/O.

    A dedicated queue keeps IMU traffic out of the two-slot image preview queue.
    Backpressure retains up to 2000 recent samples; older preview samples may be
    discarded, with a NaN break. The original capture/archive is never changed.
    """
    def __init__(self, pending, role, origin, *, interval=.1, capacity=2000):
        if interval <= 0 or capacity < 2:
            raise ValueError('Invalid IMU preview limits')
        self.queue, self.role, self.origin = pending, role, origin
        self.interval, self.capacity = interval, capacity
        self.rows = deque()
        self.last_sample = self.last_image = -math.inf
        self.clock_offset = None
        self.last_flush = -math.inf
        self.missing = False
        self.disabled = False
        self.stats = dict(samples=0, batches=0, duplicates=0, invalid=0,
                          dropped=0, resets=0, missing_frames=0, failures=0)

    def _append(self, row):
        if len(self.rows) >= self.capacity:
            self.rows.popleft()
            self.stats['dropped'] += 1
            # Keep a discontinuity instead of drawing across discarded data.
            if self.rows:
                first = self.rows[0]
                self.rows[0] = (*first[:2], *([math.nan] * (len(first)-2)))
        self.rows.append(row)

    def capture(self, pair):
        if self.disabled:
            return
        try:
            self._capture(pair)
        except Exception as exc:
            # Visualization errors must never stop the durable camera/archive.
            self.disabled = True
            self.stats['failures'] += 1
            print(f'[imu-preview] {self.role}: disabled: {type(exc).__name__}: {exc}',
                  file=sys.stderr, flush=True)

    def _capture(self, pair):
        received = float(pair.received)
        if not math.isfinite(received):
            self.stats['invalid'] += 1
            return
        host = received - self.origin
        metadata = pair.metadata or {}
        image = metadata.get('image_timestamp_seconds')
        reset = False
        if image is not None and math.isfinite(float(image)):
            image = float(image)
            if image < self.last_image:
                self.last_sample = -math.inf
                self.clock_offset = None
                self.stats['resets'] += 1
                reset = True
            self.last_image = image
            if self.clock_offset is None:
                self.clock_offset = host - image
        else:
            image = None
        valid = 0
        for sample in metadata.get('imu_samples', ()) if image is not None else ():
            try:
                device = float(sample['timestamp_seconds'])
                acc, gyro = sample[FIELDS[0]], sample[FIELDS[1]]
                if sample.get('accepted', True) is False or len(acc) != 3 or len(gyro) != 3:
                    raise ValueError('Invalid IMU vector')
                values = tuple(float(v) for v in (*acc, *gyro))
                if not all(map(math.isfinite, (device, *values))) or abs(device - image) > .5:
                    raise ValueError('Invalid IMU sample')
            except (KeyError, TypeError, ValueError, OverflowError):
                self.stats['invalid'] += 1
                continue
            valid += 1
            if device <= self.last_sample:
                self.stats['duplicates'] += 1
                continue
            self.last_sample = device
            stamp = device + self.clock_offset
            if stamp < 0:
                continue
            if reset:
                self._append((max(0., stamp - 1e-9), device, *([math.nan] * 6)))
                reset = False
            self._append((stamp, device, *values))
            self.stats['samples'] += 1
        if not valid:
            self.stats['missing_frames'] += 1
            if not self.missing:
                stamp = host if image is None else image + self.clock_offset
                self._append((max(0., stamp), math.nan, *([math.nan] * 6)))
                print(f'[imu-preview] {self.role}: no valid IMU samples; graph left empty.', flush=True)
            self.missing = True
        elif self.missing:
            self.missing = False
            print(f'[imu-preview] {self.role}: IMU samples resumed.', flush=True)
        if received - self.last_flush >= self.interval:
            self.flush(received)

    def flush(self, received=None):
        if self.disabled or not self.rows:
            return
        now = float(received) if received is not None else self.origin + self.rows[-1][0]
        self.last_flush = now
        # Detach an owned array before the multiprocessing feeder serializes it.
        packet = self._packet(now)
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            # Retry next interval, keeping buffered measurements intact.
            return
        except Exception as exc:
            self.disabled = True
            self.stats['failures'] += 1
            print(f'[imu-preview] {self.role}: queue disabled: {exc}', file=sys.stderr, flush=True)
            return
        self.rows.clear()
        self.stats['batches'] += 1


    def _packet(self, now):
        return dict(role=self.role, imu=np.asarray(self.rows, dtype=np.float64), received=now)


class RawImuLogger:
    """Viewer-owner batched scalar logging; never affects image/overlay times."""
    def __init__(self, recording, roles):
        from .imu_diagnostics import Continuity
        self.recording, self.roles = recording, frozenset(roles)
        self.continuity = {role:Continuity() for role in self.roles}
        self.upstream = {}
        for role in self.roles:
            for field in FIELDS:
                recording.log(f'/imu/{role}/{field}', rr.SeriesLines(
                    colors=COLORS, names=['X', 'Y', 'Z'], widths=1.5), static=True)

    def consume(self, packet):
        role, rows = packet['role'], np.asarray(packet['imu'], dtype=np.float64)
        if role not in self.roles or rows.ndim != 2 or rows.shape[1] != 8:
            raise ValueError('Invalid IMU preview packet')
        if not len(rows):
            return
        if not np.isfinite(rows[:, 0]).all() or (rows[:, 0] < 0).any():
            raise ValueError('Invalid IMU preview timestamps')
        # Explicit timestamps preserve all sample intervals; batches are never
        # stamped at inference completion or replaced with latest-frame time.
        indexes = [rr.TimeColumn(TIMELINE, duration=rows[:, 0])]
        lengths = np.full(len(rows), 3, dtype=np.int32)
        for group, field in enumerate(FIELDS):
            # Three scalar instances at each timestamp keep XYZ together and
            # reduce Arrow/gRPC setup from six calls per batch to two.
            self.recording.send_columns(f'/imu/{role}/{field}', indexes=indexes,
                columns=rr.Scalars.columns(scalars=rows[:, 2 + group*3:5 + group*3].ravel()).partition(lengths))
        # Only count a batch after both plots were submitted successfully.
        tracker = self.continuity[role]
        tracker.set_epoch(packet.get('clock', {}).get('clock_epoch', 0))
        for row in rows:
            tracker.row(row)
        self.upstream[role] = packet.get('raw_diagnostics', {})

    def metrics(self):
        return {role:dict(**self.upstream.get(role, {}), viewer_submitted=tracker.snapshot())
                for role,tracker in self.continuity.items()}
