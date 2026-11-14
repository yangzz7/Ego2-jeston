"""SE(3) anchoring and strict pose validity; display predictions never enter here."""
from collections import deque
from dataclasses import dataclass, field
import math
import cv2
import numpy as np
from ..common.config import se3


def relative_delta(reference, target):
    """Position difference and active relative rotation, both expressed in G.

    This is distinct from inv(T_G_reference) @ T_G_target, whose translation
    and rotation coordinates are expressed in the reference optical frame.
    """
    rotation = target[:3,:3] @ reference[:3,:3].T
    horizontal = np.hypot(rotation[0,0], rotation[1,0])
    pitch = np.arctan2(-rotation[2,0], horizontal)
    if horizontal < 1e-8:
        roll, yaw = np.arctan2(-rotation[1,2], rotation[1,1]), 0.
    else:
        roll = np.arctan2(rotation[2,1], rotation[2,2])
        yaw = np.arctan2(rotation[1,0], rotation[0,0])
    return dict(delta_p_m=(target[:3,3]-reference[:3,3]).tolist(),
                delta_R_shared=rotation.tolist(),
                delta_rpy_deg=np.rad2deg([roll,pitch,yaw]).tolist())


def inverse(t):
    t = se3(t)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def angle(a, b):
    return float(np.linalg.norm(cv2.Rodrigues(a.T @ b)[0]))


def mean_transform(values):
    values = np.asarray(values)
    u, _, vt = np.linalg.svd(values[:, :3, :3].sum(axis=0))
    out = np.eye(4)
    out[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    out[:3, 3] = np.mean(values[:, :3, 3], axis=0)
    return out


def robust_fit(values, cfg):
    """Medoid seed prevents a few remote outliers from dragging the mean away."""
    values = np.asarray(values)
    position = np.median(values[:, :3, 3], axis=0)
    distances = np.array([[angle(a[:3,:3], b[:3,:3]) for b in values] for a in values])
    centre = values[np.argmin(np.median(distances, axis=1))].copy()
    centre[:3, 3] = position
    mask = np.ones(len(values), bool)
    for _ in range(4):
        trans = np.linalg.norm(values[:, :3, 3]-centre[:3, 3], axis=1)
        rot = np.array([math.degrees(angle(centre[:3,:3], t[:3,:3])) for t in values])
        mask = (trans <= cfg['outlier_translation_m']) & (rot <= cfg['outlier_rotation_deg'])
        if not mask.any():
            return None, dict(reason='no_consensus', inliers=0)
        centre = mean_transform(values[mask])
    trans = np.linalg.norm(values[mask,:3,3]-centre[:3,3], axis=1)
    rot = np.array([math.degrees(angle(centre[:3,:3], t[:3,:3])) for t in values[mask]])
    quality = dict(inliers=int(mask.sum()), total=len(values),
                   translation_rms_m=float(np.sqrt(np.mean(trans**2))),
                   rotation_rms_deg=float(np.sqrt(np.mean(rot**2))))
    valid = (mask.sum() >= cfg['min_samples'] and mask.mean() >= cfg['min_inlier_fraction']
             and quality['translation_rms_m'] <= cfg['max_translation_rms_m']
             and quality['rotation_rms_deg'] <= cfg['max_rotation_rms_deg'])
    quality['reason'] = 'accepted' if valid else 'unstable_alignment'
    return centre if valid else None, quality


class AlignmentWindow:
    def __init__(self, cfg, board, reprojection_limit):
        self.cfg, self.board, self.reprojection_limit = cfg, board, reprojection_limit
        self.samples = deque(maxlen=120)
        self.reason = 'waiting_for_board'
        self.quality = {}
        self.resets = 0
        self.last_reset = 'none'
        self.last_reset_samples = 0
        self.expired_samples = 0

    def clear(self, reason='reset'):
        if self.samples:
            self.resets += 1
            self.last_reset = reason
            self.last_reset_samples = len(self.samples)
        self.samples.clear()
        self.reason = 'waiting_for_board'
        self.quality = {}

    def add(self, packet, measurement):
        stamp = packet['timestamp']
        if self.samples and stamp-self.samples[-1][0] > self.cfg['max_sample_gap_s']:
            self.clear('board_sample_gap')
        while self.samples and stamp-self.samples[0][0] > self.cfg['max_duration_s']:
            self.samples.popleft()
            self.expired_samples += 1
        if measurement is None:
            self.reason = 'no_valid_board_measurement'
            return None
        if not math.isfinite(measurement.reprojection_px) or measurement.reprojection_px > self.reprojection_limit:
            self.reason = 'reprojection_error'
            return None
        for camera in ('cam0', 'cam1'):
            ids = [key[1] for key in measurement.inliers if key[0] == camera]
            if len(ids) < self.cfg['min_tags_per_eye']:
                self.reason = 'insufficient_stereo_support'
                return None
            points = np.array([self.board.centres[i][:2] for i in ids])
            if np.linalg.matrix_rank(points-points.mean(axis=0)) < 2 or np.any(np.ptp(points,axis=0) < self.cfg['min_span_m']):
                self.reason = 'insufficient_board_coverage'
                return None
        if self.samples and stamp <= self.samples[-1][0]:
            self.reason = 'duplicate_or_old_frame'
            return None
        # measurement.pose is T_cam1_board; native pose is T_local_cam1.
        anchor = inverse(measurement.pose) @ inverse(packet['T_world_camera'])
        self.samples.append((stamp, anchor))
        self.reason = 'collecting_board_samples'
        if len(self.samples) < self.cfg['min_samples'] or stamp-self.samples[0][0] < self.cfg['min_duration_s']:
            return None
        result, self.quality = robust_fit([x[1] for x in self.samples], self.cfg)
        self.reason = self.quality['reason']
        return result


def interpolate(buffer, stamp, max_gap):
    for index in range(len(buffer)-1, -1, -1):
        ta, a = buffer[index]
        if abs(ta-stamp) < 1e-8:
            return a.copy()
        if ta < stamp and index+1 < len(buffer):
            tb, b = buffer[index+1]
            if stamp > tb or tb-ta > max_gap:
                return None
            f = (stamp-ta)/(tb-ta)
            out = np.eye(4)
            out[:3,3] = (1-f)*a[:3,3]+f*b[:3,3]
            delta = cv2.Rodrigues(a[:3,:3].T @ b[:3,:3])[0]
            out[:3,:3] = a[:3,:3] @ cv2.Rodrigues(f*delta)[0]
            return out
    return None


@dataclass
class DeviceState:
    window: AlignmentWindow
    packet: dict | None = None
    received: float = float('-inf')
    map_id: str | None = None
    anchor: np.ndarray | None = None
    buffer: deque = field(default_factory=lambda: deque(maxlen=120))
    board_packets: deque = field(default_factory=lambda: deque(maxlen=120))
    last_ingest: float = float('-inf')
    epoch: int = 0
    aligned_at: float | None = None
    disconnected: bool = False


class Coordinator:
    def __init__(self, cfg, devices):
        self.cfg = cfg
        self.roles = tuple(devices)
        self.reference = cfg['reference_role']
        self.devices = {role: DeviceState(AlignmentWindow(cfg['alignment'], data['board'],
                       data['aprilgrid']['quality']['max_reprojection_px'])) for role,data in devices.items()}
        self.started = False
        self.start_time = None
        self.initial_relatives = {}
        self.initial_deltas = {}
        self.initial_poses = {}
        self.shared_from_board = None
        self.events = []

    def ingest(self, role, packet, now):
        state = self.devices[role]
        if state.map_id is not None and state.map_id != packet['map_id']:
            state.anchor = None
            state.aligned_at = None
            state.window.clear('map_changed')
            state.buffer.clear()
            state.board_packets.clear()
            state.epoch += 1
            self.events.append(dict(event='alignment_invalidated', role=role, time=packet['capture_time'], reason='map_changed'))
        # A valid result after a silent transport gap must not resurrect a
        # board job from before that gap, even when the map ID is unchanged.
        previous = state.packet
        stale = self.cfg['timing']['stale_s']
        if previous is not None and (now-state.last_ingest > stale or
                packet['capture_time']-previous['capture_time'] > stale):
            state.buffer.clear()
            state.board_packets.clear()
            state.window.clear('vio_stream_gap')
        state.last_ingest = now
        state.map_id = packet['map_id']
        state.packet = packet
        state.received = min(now, packet.get('host_monotonic', now))
        state.disconnected = False
        if packet['valid'] and packet['initialization_complete']:
            pose = se3(packet['T_world_camera'])
            state.buffer.append((packet['capture_time'], pose))
            # Keep the pose paired with the original image; never substitute
            # the current pose when an asynchronous board result arrives.
            state.board_packets.append(dict(packet, T_world_camera=pose.tolist()))
        else:
            state.buffer.clear()  # Interpolation must never bridge tracking loss.
            state.board_packets.clear()
            state.window.clear('tracking_invalid')

    def ready(self, role, now):
        s = self.devices[role]
        return bool(s.packet and not s.disconnected and now-s.received <= self.cfg['timing']['stale_s']
                    and s.packet['valid'] and s.packet['initialization_complete'])

    def observe_board(self, role, packet, measurement, now):
        s = self.devices[role]
        if s.anchor is not None or packet['map_id'] != s.map_id:
            return
        if not all(self.ready(r, now) for r in self.roles):
            s.window.clear('live_vio_unavailable')
            s.window.reason = 'waiting_for_all_vio'
            return
        if not packet['valid'] or not packet['initialization_complete']:
            return
        source = next((p for p in reversed(s.board_packets)
                       if all(p[k] == packet[k] for k in ('map_id','frame_id','capture_time','timestamp'))), None)
        if source is None:
            s.window.reason = 'board_frame_not_in_current_track'
            return
        age = now-source.get('host_monotonic', now)
        budget = self.cfg['timing'].get('board_max_age_s', self.cfg['timing']['stale_s'])
        if not math.isfinite(age) or age < 0:
            s.window.reason = 'invalid_board_time'
            return
        if age > budget:
            # Reject this result only. The next accepted source timestamp
            # still enforces max_sample_gap_s and max_duration_s in add().
            s.window.reason = 'board_processing_too_old'
            return
        anchor = s.window.add(source, measurement)
        if anchor is not None:
            s.anchor, s.aligned_at = anchor, packet['capture_time']
            self.events.append(dict(event='aligned', role=role, time=s.aligned_at,
                                    T_board_local=anchor.tolist(), quality=dict(s.window.quality)))

    def relative(self, now):
        if not all(self.ready(r, now) and self.devices[r].anchor is not None and self.devices[r].buffer for r in self.roles):
            return None
        stamp = min(self.devices[r].buffer[-1][0] for r in self.roles)
        if stamp < max(self.devices[r].aligned_at for r in self.roles):
            return None
        poses = {r: interpolate(self.devices[r].buffer, stamp, self.cfg['timing']['interpolation_gap_s']) for r in self.roles}
        if any(t is None for t in poses.values()):
            return None
        board_poses = {r: self.devices[r].anchor @ poses[r] for r in self.roles}
        relatives = {r: inverse(board_poses[self.reference]) @ board_poses[r]
                     for r in self.roles if r != self.reference}
        if not self.started:
            # Reference VIO has already completed gravity alignment (+Z up).
            # Keep the board centre as origin, but use that VIO world's axes.
            # Freeze this rotation for the entire session, including map resets.
            self.shared_from_board = np.eye(4)
            self.shared_from_board[:3,:3] = self.devices[self.reference].anchor[:3,:3].T
            self.started, self.start_time = True, stamp
            self.initial_relatives = {r: t.copy() for r,t in relatives.items()}
            self.initial_poses = {r: self.shared_from_board @ t for r,t in board_poses.items()}
            self.initial_deltas = {r: relative_delta(self.initial_poses[self.reference], self.initial_poses[r])
                                   for r in relatives}
            event = dict(event='started', time=stamp, reference_role=self.reference,
                         initial_T_reference_camera={r:t.tolist() for r,t in relatives.items()},
                         initial_deltas=self.initial_deltas,
                         T_shared_board=self.shared_from_board.tolist())
            self.events.append(event)
        return stamp, relatives

    def board_pose(self, role, now):
        s = self.devices[role]
        if not self.started or s.anchor is None or not self.ready(role, now):
            return None
        return s.anchor @ se3(s.packet['T_world_camera'])

    def pose(self, role, now):
        pose = self.board_pose(role, now)
        return self.shared_from_board @ pose if pose is not None else None

    def status(self, role, now):
        s = self.devices[role]
        if s.disconnected:
            return 'DISCONNECTED'
        if s.packet is None:
            return 'CONNECTING'
        if now-s.received > self.cfg['timing']['stale_s']:
            return 'STALE'
        if not s.packet['initialization_complete']:
            return 'VIO INITIALIZING (move with translation and rotation)'
        if not s.packet['valid']:
            return 'TRACKING LOST'
        if s.anchor is None:
            return f'ALIGNING: {s.window.reason} | {len(s.window.samples)} samples'
        if not self.started:
            return 'ALIGNED | waiting for other device / common time'
        return 'RUNNING | alignment locked'
