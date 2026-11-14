"""Calibrated non-planar wrist bundles; no assumed equal spacing or wrist radius."""
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
import hashlib
import json
import os
import shlex
import tempfile
import cv2
import numpy as np
from ..common.config import se3
from ..common.paths import ROOT
from .pose import Measurement, rotation_distance


def marker_ids(target):
    return list(target.get('ids', [target.get('id')]))


def mean_pose(poses):
    result = np.eye(4)
    u, _, vt = np.linalg.svd(np.sum([p[:3, :3] for p in poses], axis=0))
    result[:3, :3] = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    result[:3, 3] = np.mean([p[:3, 3] for p in poses], axis=0)
    return result


def delta(a, b):
    return np.linalg.norm(a[:3, 3]-b[:3, 3]), np.rad2deg(rotation_distance(a[:3, :3], b[:3, :3]))


def consensus(poses, distance, angle, fraction=0.7):
    """Largest mutually close cluster, requiring a unique strong majority."""
    positions = np.array([p[:3, 3] for p in poses])
    rotations = np.array([p[:3, :3] for p in poses])
    translation = np.linalg.norm(positions[:, None]-positions[None, :], axis=-1)
    cosine = (np.einsum('aij,bij->ab', rotations, rotations)-1)/2
    angles = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    near = (translation <= distance) & (angles <= angle)
    index = int(np.argmax(near.sum(axis=1)))
    mask = near[index]
    if mask.sum() < fraction*len(poses):
        raise ValueError('No stable majority; wrist geometry may be changing')
    selected = [p for p, ok in zip(poses, mask) if ok]
    result = mean_pose(selected)
    # Also require each selected measurement to agree with the final mean.
    selected = [p for p in selected if delta(p, result)[0] <= distance and delta(p, result)[1] <= angle]
    if len(selected) < fraction*len(poses):
        raise ValueError('Unstable relative marker poses')
    return mean_pose(selected), len(selected)


def wrist_frame(poses, ids, palm_offset_deg=0):
    """Fit marker-centre ellipse; seam is an approximate palm-direction reference."""
    points = np.array([poses[i][:3, 3] for i in ids])
    centroid = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points-centroid)
    if singular[1] < .008 or singular[2]/singular[1] > .3:
        raise ValueError('Tags must form a single approximately planar loop around the wrist')
    axes = vt[:2].T
    xy = (points-centroid) @ axes
    (cx, cy), diameters, _ = cv2.fitEllipse(xy.astype(np.float32).reshape(-1, 1, 2))
    if not .025 < min(diameters) <= max(diameters) < .16 or min(diameters)/max(diameters) < .3:
        raise ValueError('Implausible wrist ellipse; collect more accurate overlapping observations')
    centre = centroid + axes @ np.array([cx, cy])
    offsets = xy-np.array([cx, cy])
    angles = np.sort(np.arctan2(offsets[:, 1], offsets[:, 0]))
    max_gap = np.rad2deg(np.diff(np.r_[angles, angles[0]+2*np.pi])).max()
    if max_gap > 115:
        raise ValueError('Calibrated tags do not cover a complete wrist loop')
    seam = (points[0]+points[-1])/2
    z = seam-centre
    z -= vt[2]*(z @ vt[2])
    if np.linalg.norm(z) < .008:
        raise ValueError('Cannot determine seam/palm direction')
    z /= np.linalg.norm(z)
    x = points[0]-points[-1]  # last ID -> first ID across the seam
    x -= vt[2]*(x @ vt[2])
    x -= z*(x @ z)
    if np.linalg.norm(x) < .005:
        raise ValueError('Seam markers are coincident or unordered')
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    # Check the supplied ID order is consistent around a single loop.
    relative = points-centre
    a = np.arctan2(relative @ x, relative @ z)
    increments = (np.diff(np.r_[a, a[0]])+2*np.pi) % (2*np.pi)
    if abs(increments.sum()-2*np.pi) > .1 or np.max(increments) > np.deg2rad(115):
        raise ValueError('IDs must be listed in physical circumferential order')
    result = np.eye(4)
    result[:3, :3] = np.column_stack([x, y, z]) @ cv2.Rodrigues(np.array([0., np.deg2rad(palm_offset_deg), 0.]))[0]
    result[:3, 3] = centre
    return result, {'ellipse_diameters_m': list(map(float, diameters)), 'max_angular_gap_deg': float(max_gap),
                    'plane_rms_m': float(np.sqrt(np.mean(((points-centroid) @ vt[2])**2))),
                    'origin': 'estimated wrist marker-loop centre', 'palm': 'approximate first/last-ID seam direction'}


class Calibrator:
    def __init__(self, target, settings):
        self.target, self.settings = target, settings
        self.ids = marker_ids(target)
        self.edges = defaultdict(lambda: deque(maxlen=160))
        self.last_stamp = -float('inf')

    def add(self, stamp, measurements):
        if stamp-self.last_stamp < .06:
            return
        self.last_stamp = stamp
        valid = [(i, measurements[i][0].T_head_marker) for i in self.ids
                 if i in measurements and measurements[i][0] is not None and measurements[i][0].mode == 'stereo']
        for (i, a), (j, b) in combinations(valid, 2):
            self.edges[i, j].append((stamp, np.linalg.inv(a) @ b))

    def progress(self):
        return {f'{i}-{j}': len(samples) for (i, j), samples in self.edges.items()}

    def finish(self):
        settings = self.settings
        edges = []
        for (i, j), samples in self.edges.items():
            if len(samples) < settings['min_samples'] or samples[-1][0]-samples[0][0] < settings['min_span_s']:
                continue
            try:
                pose, count = consensus([p for _, p in samples], settings['max_translation_m'], settings['max_rotation_deg'])
            except ValueError:
                continue
            if count >= settings['min_samples']:
                edges.append((i, j, pose, count))
        # A closed ring is needed to check accumulation error, including the palm seam.
        pairs = {frozenset((i, j)) for i, j, _, _ in edges}
        required = [frozenset((i, j)) for i, j in zip(self.ids, self.ids[1:]+self.ids[:1])]
        missing = [sorted(p) for p in required if p not in pairs]
        if missing:
            raise ValueError(f"{self.target['name']}: insufficient stable co-visible stereo samples for adjacent pairs {missing}")
        anchor = self.ids[0]
        poses = {anchor: np.eye(4)}
        while len(poses) < len(self.ids):
            candidates = [(i, j, t, n) for i, j, t, n in edges if (i in poses) != (j in poses)]
            if not candidates:
                raise ValueError('Disconnected calibration graph')
            i, j, t, _ = max(candidates, key=lambda e: e[3])
            if i in poses:
                poses[j] = poses[i] @ t
            else:
                poses[i] = poses[j] @ np.linalg.inv(t)
        # Relax the entire graph while keeping the anchor fixed. All accepted edges
        # contribute, so seam closure error is not assigned to a single boundary.
        for _ in range(160):
            updated = {anchor: poses[anchor]}
            for node in self.ids[1:]:
                predictions = [poses[i] @ t if node == j else poses[j] @ np.linalg.inv(t)
                               for i, j, t, _ in edges if node in (i, j)]
                updated[node] = mean_pose([poses[node], mean_pose(predictions)])
            change = max(delta(poses[i], updated[i])[0] + .02*np.deg2rad(delta(poses[i], updated[i])[1]) for i in self.ids)
            poses = updated
            if change < 1e-9:
                break
        diagnostics = []
        for i, j, t, n in edges:
            distance, angle = delta(np.linalg.inv(poses[i]) @ poses[j], t)
            if distance > settings['max_translation_m'] or angle > settings['max_rotation_deg']:
                raise ValueError('Loop closure inconsistent: collect again with the wristband held rigid')
            diagnostics.append({'pair': [i, j], 'samples': n, 'closure_m': float(distance), 'closure_deg': float(angle)})
        frame, geometry = wrist_frame(poses, self.ids, self.target.get('palm_offset_deg', 0))
        wrist_from_anchor = np.linalg.inv(frame)
        return {'ids': self.ids, 'size_m': self.target['size_m'], 'palm_offset_deg': self.target.get('palm_offset_deg', 0),
                'T_wrist_marker': {str(i): (wrist_from_anchor @ poses[i]).tolist() for i in self.ids},
                'geometry': geometry, 'edges': diagnostics}


def signature(cfg):
    return {'dictionary': cfg['dictionary'], 'border_bits': cfg['border_bits'],
            'camera_sha256': hashlib.sha256(Path(cfg['calibration']['file']).read_bytes()).hexdigest()}


def save_calibrations(path, cfg, results):
    path = Path(path)
    document = {'version': 1, **signature(cfg), 'targets': {}}
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get('version') != 1 or any(previous.get(k) != v for k, v in signature(cfg).items()):
            raise ValueError('Existing wrist calibration uses different camera/tag settings; choose a new --bundle-calibration file')
        document['targets'].update(previous['targets'])
    document['targets'].update(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.wrist-calibration-', delete=False) as f:
        temporary = Path(f.name)
        json.dump(document, f, indent=2)
        f.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_bundles(cfg):
    path = Path(cfg['bundle_calibration'])
    if not path.exists():
        command = [str(ROOT / 'run_hand_pose.sh'), 'calibrate']
        if cfg.get('_path'):
            command += ['--config', cfg['_path']]
        if cfg.get('input', {}).get('serial'):
            command += ['--sn', cfg['input']['serial']]
        command += ['--bundle-calibration', str(path), '--hand', 'left']
        raise ValueError(f'需要先标定腕带，缺少文件 {path}：{shlex.join(command)}；然后将 --hand 改为 right')
    document = json.loads(path.read_text())
    if document.get('version') != 1 or any(document.get(k) != v for k, v in signature(cfg).items()):
        raise ValueError('Wrist calibration does not match the camera/dictionary/border settings')
    for target in cfg['targets']:
        if 'ids' not in target:
            continue
        entry = document['targets'].get(target['name'])
        if (entry is None or entry['ids'] != target['ids'] or abs(entry['size_m']-target['size_m']) > 1e-9):
            raise ValueError(f"{target['name']}: no matching wrist calibration; calibrate this hand again")
        transforms = entry['T_wrist_marker']
        if set(transforms) != {str(i) for i in target['ids']}:
            raise ValueError('Incomplete wrist marker geometry')
        correction = np.eye(4)
        correction[:3, :3] = cv2.Rodrigues(np.array([0., np.deg2rad(entry.get('palm_offset_deg', 0)-target.get('palm_offset_deg', 0)), 0.]))[0]
        target['T_wrist_marker'] = {k: (correction @ se3(v)).tolist() for k, v in transforms.items()}
        target['bundle_diagnostics'] = entry.get('geometry', {})


def fuse(target, measurements, settings):
    candidates, ids, errors, modes = [], [], [], []
    for marker_id in marker_ids(target):
        measurement, _ = measurements[marker_id]
        if measurement is None:
            continue
        candidates.append(measurement.T_head_marker @ np.linalg.inv(np.asarray(target['T_wrist_marker'][str(marker_id)])))
        ids.append(marker_id); errors.append(measurement.reprojection_px); modes.append(measurement.mode)
    if not candidates:
        return None, 'no_valid_bundle_markers'
    if len(candidates) > 1:
        # Deliberately reject conflicting tags, instead of selecting an arbitrary
        # side when only two tags disagree (often a shifted or bending wristband).
        for a, b in combinations(candidates, 2):
            distance, angle = delta(a, b)
            if distance > settings['max_translation_m'] or angle > settings['max_rotation_deg']:
                return None, 'bundle_geometry_changed'
    result = Measurement(mean_pose(candidates), float(max(errors)),
                         'bundle_stereo' if all(m == 'stereo' for m in modes) else 'bundle_mono', tuple(ids))
    return result, 'ok'


def synthetic_bundle(target):
    """Idealized demo-only ring, never used for a real wrist without calibration."""
    result = {}
    ids = marker_ids(target)
    for index, marker_id in enumerate(ids):
        angle = 2*np.pi*(index+.5)/len(ids)
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([0., angle, 0.]))[0]
        pose[:3, 3] = [.034*np.sin(angle), 0, .028*np.cos(angle)]
        result[str(marker_id)] = pose.tolist()
    return result
