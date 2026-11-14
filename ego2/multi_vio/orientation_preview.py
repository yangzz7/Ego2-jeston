"""Derived Mahony attitude preview; raw IMU recording stays untouched."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import rerun as rr
import rerun.blueprint as b

from ..common.config import read_yaml, se3
from ..common.mahony import Mahony, Settings
from ..common.viewer import TIMELINE
from .raw_imu import RawImuPublisher

COLORS = ([235, 235, 235], [239, 95, 98], [89, 204, 124], [86, 155, 255])


def charts(role, label):
    from .raw_imu import charts as raw_charts
    from .body_preview import view
    base = f'/imu/{role}/orientation'
    quaternion = b.TimeSeriesView(name='Quaternion (cam0 relative)',
        origin=base+'/quaternion_wxyz', contents='$origin/**',
        axis_x=b.TimeAxis(view_range=b.TimeRange(
            b.TimeRangeBoundary.cursor_relative(seconds=-10),
            b.TimeRangeBoundary.cursor_relative(seconds=0))),
        axis_y=b.ScalarAxis(range=[-1.05, 1.05], zoom_lock=True),
        plot_legend=b.PlotLegend(corner='RightTop', visible=True))
    return b.Vertical(
        b.Horizontal(quaternion, *raw_charts(role, ''), view(role,label), column_shares=[1,1,1,1]),
        b.TextDocumentView(name='Attitude / hand source', origin=base+'/body_status', contents='$origin'),
        row_shares=[5,1.3])


class QuaternionPublisher(RawImuPublisher):
    """Fuse each accepted device sample BEFORE any disposable display queue.

    The inherited observer validates/anchors timestamps and bounds preview
    memory. Packet 'imu' now holds [capture_time, device_time, qw, qx, qy, qz].
    Status transitions are timestamped separately, without changing image time.
    """
    def __init__(self, pending, role, origin, profile, **kwargs):
        super().__init__(pending, role, origin, **kwargs)
        self.events = []
        self.last_status = None
        self.body_reference = None
        self.estimator = None
        self.calibration_error = None
        self.meta = dict(algorithm='Mahony', ki=0., quaternion_order='wxyz',
            rotation='current cam0 -> initial cam0', heading='relative; unobserved drift',
            gap_policy='15..150 ms: endpoint-rate estimate, quality flag; >150 ms: reset reference',
            clock='fixed device-to-host offset; not hardware synchronization')
        try:
            calibration = profile['calibration']
            if calibration.get('format', 'kalibr') != 'kalibr':
                raise ValueError('Kalibr T_cam_imu required')
            path = Path(calibration['file'])
            r_ci = se3(read_yaml(path)['cam0']['T_cam_imu'])[:3,:3]
            settings = Settings(gravity=float(profile.get('input', {}).get('acceleration_scale', 9.80665)))
            self.estimator = Mahony(r_ci, settings)
            self.meta.update(calibration_file=str(path), calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             camera_from_imu_rotation=r_ci.tolist(), settings=asdict(settings))
        except Exception as exc:
            self.calibration_error = f'CALIBRATION ERROR: cam0 T_cam_imu unavailable ({exc})'
            self.meta['error'] = self.calibration_error
        self.stats.update(valid_quaternions=0, reference_segments=0, attitude_resets=0, sparse_imu_gaps=0,
                          fusion_capture_ms_total=0., fusion_capture_ms_max=0.)

    def _capture(self, pair):
        started = time.perf_counter()
        super()._capture(pair)
        elapsed = (time.perf_counter()-started)*1000
        self.stats['fusion_capture_ms_total'] += elapsed
        self.stats['fusion_capture_ms_max'] = max(self.stats['fusion_capture_ms_max'], elapsed)

    def _append(self, row):
        stamp, device, *values = row
        q = None
        if self.estimator is not None:
            if math.isfinite(device) and all(map(math.isfinite, values)):
                q = self.estimator.update(device, values[:3], values[3:])
            else:
                self.estimator.reset()
            status = self.estimator.status
            self.stats.update(valid_quaternions=self.estimator.valid_samples,
                reference_segments=self.estimator.segment, attitude_resets=self.estimator.resets,
                sparse_imu_gaps=self.estimator.sparse_gaps)
        else:
            status = self.calibration_error
        if status != self.last_status:
            print(f'[attitude] {self.role}: {status}', flush=True)
            self.events.append((stamp, status))
            self.events = self.events[-100:]
            self.last_status = status
        super()._append((stamp, device, *(q if q is not None else [math.nan]*4)))
        if q is not None:
            self._body_reference(stamp, self.estimator.segment)
        else:
            self.body_reference = None

    def _body_reference(self, stamp, segment):
        """Publish the existing gravity reference, without changing fusion.

        It maps initial cam0 to Mahony's gravity-aligned Z-up world. A start
        timestamp prevents applying a rebuilt reference to older samples.
        """
        if self.body_reference is not None and self.body_reference['segment'] == int(segment):
            return
        reference = self.estimator.reference
        if reference is not None:
            self.body_reference = dict(segment=int(segment), start=float(stamp),
                                       world_from_initial_camera_wxyz=np.asarray(reference).tolist())

    def _packet(self, now):
        packet = super()._packet(now)
        packet.update(orientation=True, status=list(self.events), metadata=self.meta,
                      body_reference=self.body_reference)
        return packet

    def flush(self, received=None):
        batches = self.stats['batches']
        super().flush(received)
        if self.stats['batches'] != batches:
            self.events.clear()


class QuaternionLogger:
    """One four-instance scalar batch; status uses explicit column timestamps."""
    def __init__(self, recording, roles):
        self.recording, self.roles = recording, frozenset(roles)
        self.metadata_logged = set()
        for role in self.roles:
            path = f'/imu/{role}/orientation'
            recording.log(path+'/quaternion_wxyz', rr.SeriesLines(
                colors=COLORS, names=['qw', 'qx', 'qy', 'qz'], widths=1.5), static=True)
            recording.send_columns(path+'/status', indexes=[rr.TimeColumn(TIMELINE, duration=[0.])],
                columns=rr.TextDocument.columns(text=['WAITING: hold camera still for 2.5 s']))

    def consume(self, packet):
        role, rows = packet['role'], np.asarray(packet['imu'], dtype=np.float64)
        if role not in self.roles or not packet.get('orientation') or rows.ndim != 2 or rows.shape[1] != 6:
            raise ValueError('Invalid attitude preview packet')
        if not len(rows):
            return
        if not np.isfinite(rows[:,0]).all() or (rows[:,0] < 0).any():
            raise ValueError('Invalid attitude timestamps')
        path = f'/imu/{role}/orientation'
        if role not in self.metadata_logged:
            self.recording.log(path+'/metadata', rr.TextDocument(json.dumps(packet['metadata'], ensure_ascii=False)), static=True)
            self.metadata_logged.add(role)
        self.recording.send_columns(path+'/quaternion_wxyz',
            indexes=[rr.TimeColumn(TIMELINE, duration=rows[:,0])],
            columns=rr.Scalars.columns(scalars=rows[:,2:].ravel()).partition(np.full(len(rows), 4, dtype=np.int32)))
        events = packet.get('status', [])
        if events:
            self.recording.send_columns(path+'/status',
                indexes=[rr.TimeColumn(TIMELINE, duration=[e[0] for e in events])],
                columns=rr.TextDocument.columns(text=[e[1] for e in events]))

    def unavailable(self, role, stamp, text):
        self.consume(dict(role=role, orientation=True,
            imu=np.array([[max(0.,stamp), math.nan, math.nan, math.nan, math.nan, math.nan]]),
            metadata=dict(error=text), status=[(max(0.,stamp),text)]))
