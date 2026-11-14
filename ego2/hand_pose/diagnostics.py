"""Explain where calibration stopped, without weakening geometric checks."""
from collections import Counter, defaultdict
from .bundle import marker_ids


class ObservationDiagnostics:
    def __init__(self, cfg):
        self.targets = {t['name']: marker_ids(t) for t in cfg['targets']}
        self.cameras = [c['camera'] for c in cfg['channels']]
        self.frames = 0
        self.duration = 0.
        self.decoded = {camera: Counter() for camera in self.cameras}
        self.valid = Counter()
        self.rejected = defaultdict(Counter)
        self.last_seen = []
        self.last_valid = []
        self.first_time = None
        self.max_simultaneous_valid = {name: 0 for name in self.targets}

    def update(self, stamp, seen, measurements):
        if self.first_time is None:
            self.first_time = stamp
        self.duration = stamp-self.first_time
        self.frames += 1
        self.last_seen = list(seen)
        self.last_valid = [i for i, (measurement, _) in measurements.items() if measurement is not None]
        for item in seen:
            self.decoded[item['camera']][int(item['id'])] += 1
        for marker_id, (measurement, reason) in measurements.items():
            if measurement is not None:
                self.valid[marker_id] += 1
            elif reason != 'not_seen':
                self.rejected[marker_id][reason] += 1
        for name, ids in self.targets.items():
            self.max_simultaneous_valid[name] = max(self.max_simultaneous_valid[name], len(set(ids) & set(self.last_valid)))

    def summary(self):
        seen = set().union(*(set(c) for c in self.decoded.values()))
        targets = {}
        for name, ids in self.targets.items():
            visible = sorted(set(ids) & seen)
            valid = {i: self.valid[i] for i in ids if self.valid[i]}
            reasons = Counter()
            for i in ids:
                reasons.update(self.rejected[i])
            if not visible:
                state = 'no_target_decoded'
                hint = '没有识别到本侧标签；先检查标签平整度、清晰度、黑边和编号。不要继续盲等标定。'
            elif not valid:
                state = 'no_valid_pose'
                if reasons and reasons.most_common(1)[0][0] == 'need_both_cameras':
                    hint = '已识码，但同一个码没有同时被两眼可靠识别；让完整标签在两幅画面中清晰可见。'
                else:
                    hint = '已识码但位姿检查未通过；检查标签是否弯曲、实际尺寸及相机标定，不能通过放宽阈值强行通过。'
            elif self.max_simultaneous_valid[name] < 2:
                state = 'no_coview_pair'
                hint = '已有单个标签的有效位姿，但没有同时得到两个标签的位姿；腕带标定需要共视标签对。'
            else:
                state = 'collecting_pairs'
                hint = '已有有效共视；继续展示缺失标签对和首尾接缝，保持腕带形状固定。'
            targets[name] = {'state': state, 'hint': hint, 'decoded_ids': visible, 'never_decoded_ids': sorted(set(ids)-seen),
                             'valid_pose_counts': valid, 'rejected_pose_counts': dict(reasons),
                             'max_simultaneous_valid': self.max_simultaneous_valid[name]}
        return {'frames': self.frames, 'duration_s': self.duration,
                'decoded_by_camera': {name: dict(counts) for name, counts in self.decoded.items()}, 'targets': targets}

    def line(self):
        cameras = ' | '.join(f"{camera}={sorted({int(x['id']) for x in self.last_seen if x['camera'] == camera})}" for camera in self.cameras)
        status = ' | '.join(f"{name}: {entry['state']}，累计有效码={list(entry['valid_pose_counts'])}，未识别={entry['never_decoded_ids']}"
                            for name, entry in self.summary()['targets'].items())
        return f"{cameras} | 本帧有效={self.last_valid} | {status}"

    def stalled(self, seconds=15):
        return self.duration >= seconds and not self.valid
