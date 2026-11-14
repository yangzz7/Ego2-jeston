"""Short-lived image-space hand IDs, time-based smoothing and label hysteresis."""
from collections import Counter, deque
from dataclasses import dataclass, field
import itertools
import math

import numpy as np

from .recognizer import Detection

IDENTITY_HOLD_S = .65


class StableValue:
    def __init__(self, initial):
        self.value = self.candidate = initial
        self.since = 0.0

    def update(self, value, stamp, hold):
        if value != self.candidate:
            self.candidate, self.since = value, stamp
        if stamp - self.since >= hold:
            self.value = value
        return self.value


class StableHandedness:
    """Confirm labels per track; never delay landmarks or alter model output."""
    def __init__(self):
        self.value = "Unknown"
        self.support_stamp = -math.inf
        self.support_score = 0.
        self.reset_pending()

    def reset_pending(self):
        self.candidate = None
        self.since = None
        self.samples = 0
        self.initial_evidence = deque(maxlen=128)

    def update(self, label, confidence, stamp, settings):
        valid = label in ("Left", "Right") and math.isfinite(confidence) and 0 <= confidence <= 1
        if self.value == "Unknown":
            # Weak predictions are neutral during initial acquisition. Expire
            # old evidence, and restart on a reliable opposite prediction.
            window = settings["handedness_initial_window_ms"] / 1000
            while self.initial_evidence and stamp - self.initial_evidence[0] > window:
                self.initial_evidence.popleft()
            if not valid or confidence < settings["handedness_confidence"]:
                return self.value
            if label != self.candidate:
                self.initial_evidence.clear()
                self.candidate = label
            self.initial_evidence.append(stamp)
            hold = settings["handedness_confirm_ms"] / 1000
            if len(self.initial_evidence) >= 3 and stamp - self.initial_evidence[0] >= hold:
                self.value = label
                self.support_stamp, self.support_score = stamp, confidence
                self.reset_pending()
            return self.value

        if valid and label == self.value and confidence >= settings['handedness_confidence']:
            self.support_stamp, self.support_score = stamp, confidence
        # Once confirmed, require continuous strong contrary evidence. Weak
        # frames break a pending switch but cannot overwrite the stable label.
        if (not valid or confidence < settings["handedness_switch_confidence"]
                or label == self.value):
            self.reset_pending()
            return self.value
        if label != self.candidate:
            self.candidate, self.since, self.samples = label, stamp, 1
        else:
            self.samples += 1
        hold = settings["handedness_switch_ms"] / 1000
        if self.samples >= 3 and stamp - self.since >= hold:
            self.value = label
            self.support_stamp, self.support_score = stamp, confidence
            self.reset_pending()
        return self.value

    def evidence(self, label, confidence, stamp, settings):
        """Current observation's identity evidence, separate from raw model score.

        A brief weak result may use a confirmed track's recent support. Medium
        or strong opposite evidence is withheld from 3D until resolved. This
        never supplies landmarks for a missing detection.
        """
        age = stamp-self.support_stamp
        if self.value not in ('Left','Right'):
            status = 'unconfirmed'
        elif label not in ('Left','Right','Unknown') or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            status = 'invalid'
        elif label in ('Left','Right') and label != self.value and confidence >= settings['handedness_confidence']:
            status = 'conflict'
        elif age > IDENTITY_HOLD_S:
            status = 'expired'
        else:
            status = 'confirmed' if age <= 1e-9 else 'held'
        return dict(side=self.value, status=status, support_score=self.support_score,
                    support_threshold=settings['handedness_confidence'],
                    support_age_s=float(age) if math.isfinite(age) else None)


def usable_side_evidence(evidence, side):
    """Validate the derived identity contract without rewriting raw scores."""
    if not isinstance(evidence, dict) or evidence.get('side') != side or side not in ('Left','Right'):
        return False
    try:
        age, score = float(evidence['support_age_s']), float(evidence['support_score'])
        threshold = float(evidence['support_threshold'])
    except (KeyError, TypeError, ValueError):
        return False
    return (evidence.get('status') in ('confirmed','held') and 0 <= age <= IDENTITY_HOLD_S
            and 0 <= threshold <= score <= 1. and math.isfinite(age) and math.isfinite(score))


@dataclass
class Track:
    id: int
    points: np.ndarray
    last: float
    handedness: str = "Unknown"
    handedness_state: StableHandedness = field(default_factory=StableHandedness)
    missing: bool = False
    observed_points: np.ndarray | None = None
    gesture: StableValue = field(default_factory=lambda: StableValue("Unknown"))
    pinch: StableValue = field(default_factory=lambda: StableValue(False))
    released_at: float = -math.inf


@dataclass
class Hand:
    id: int
    points: np.ndarray
    visible: np.ndarray
    handedness: str
    gesture: str
    score: float | None
    pinch: str
    pinch_ratio: float | None
    identity: dict | None = None


def palm_center(points):
    return points[[0, 5, 9, 13, 17], :2].mean(axis=0)


def pinch_ratio(points, visible, aspect):
    if not visible[[0, 4, 5, 8, 9, 17]].all():
        return None
    xy = points[:, :2] * [aspect, 1.0]
    size = max(np.linalg.norm(xy[0] - xy[9]), np.linalg.norm(xy[5] - xy[17]))
    if size < 1e-4:
        return None
    return float(np.linalg.norm(xy[4] - xy[8]) / size)


class HandTracker:
    def __init__(self, settings, aspect=1.0):
        from .config import DEFAULTS
        self.cfg, self.aspect = {**DEFAULTS, **settings}, aspect
        self.tracks = {}
        self.next_id = 1
        self.last_stamp = -math.inf
        self.stats = Counter()
        self.events = []

    def event(self, reason, track_id, **fields):
        self.stats[reason] += 1
        self.events.append(dict(reason=reason, track_id=track_id, **fields))

    def _recovery_gate(self, detection, track, visible, distance):
        """Extended grace is association-only and requires affirmative evidence."""
        if track.handedness not in ('Left','Right'):
            return 'unconfirmed_identity'
        if (detection.handedness != track.handedness or
                not math.isfinite(detection.handedness_score) or
                not self.cfg['handedness_confidence'] <= detection.handedness_score <= 1):
            return 'identity_conflict'
        if distance > min(.12, self.cfg['association_distance']):
            return 'distance'
        palm = [0,5,9,13,17]
        if not np.asarray(visible, dtype=bool)[palm].all():
            return 'palm_visibility'
        old = track.observed_points if track.observed_points is not None else track.points
        a, b = old[palm,:2], detection.points[palm,:2]
        a, b = a-a.mean(0), b-b.mean(0)
        size_a, size_b = np.linalg.norm(a), np.linalg.norm(b)
        if not np.isfinite(np.r_[a.ravel(),b.ravel()]).all() or min(size_a,size_b) < 1e-5:
            return 'palm_geometry'
        if not .65 <= size_b/size_a <= 1.5 or np.sqrt(np.mean((a/size_a-b/size_b)**2)) > .20:
            return 'palm_geometry'
        return None

    def update(self, detections: list[Detection], stamp, visibility=None):
        if not math.isfinite(stamp) or stamp <= self.last_stamp:
            raise ValueError("Tracking timestamps must be finite and strictly increasing")
        self.last_stamp = stamp
        self.events = []
        if len(detections) > 2:
            raise ValueError("At most two hands per eye are supported")
        if visibility is None:
            visibility = [np.ones(21, dtype=bool) for _ in detections]
        if len(visibility) != len(detections):
            raise ValueError("Visibility must match detections")
        timeout = self.cfg["track_timeout_ms"] / 1000
        recovery = self.cfg['track_reacquire_ms'] / 1000
        for tid,t in list(self.tracks.items()):
            lifetime = recovery if t.handedness in ('Left','Right') else timeout
            if stamp-t.last > lifetime + 1e-9:
                self.event('track_expired',tid,gap_s=stamp-t.last,side=t.handedness)
                del self.tracks[tid]
        ids = list(self.tracks)
        gate = self.cfg["association_distance"]
        costs = {}
        for index,(d,visible) in enumerate(zip(detections,visibility)):
            for tid in ids:
                t = self.tracks[tid]; gap = stamp-t.last
                distance = float(np.linalg.norm(palm_center(d.points)-palm_center(t.points)))
                reason = 'distance' if not math.isfinite(distance) or distance > gate else None
                if reason is None and gap > timeout + 1e-9:
                    reason = self._recovery_gate(d,t,visible,distance)
                if reason is not None:
                    self.event('association_'+reason,tid,gap_s=gap)
                    costs[index,tid] = math.inf
                    continue
                cost = distance
                if (t.handedness in ('Left','Right') and d.handedness in ('Left','Right')
                        and d.handedness != t.handedness and d.handedness_score >= .8):cost += .05
                costs[index,tid] = cost
        # After a gap, two equally plausible matches must not inherit identity.
        ambiguous = []
        for index in range(len(detections)):
            for tid in ids:
                if stamp-self.tracks[tid].last <= timeout + 1e-9 or not math.isfinite(costs[index,tid]):continue
                cost = costs[index,tid]
                alternatives = [costs[index,other] for other in ids if other!=tid]
                alternatives += [costs[other,tid] for other in range(len(detections)) if other!=index]
                if any(abs(other-cost)<.03 for other in alternatives):
                    self.event('association_ambiguous',tid,gap_s=stamp-self.tracks[tid].last)
                    ambiguous.append((index,tid))
        for pair in ambiguous:
            costs[pair] = math.inf
        # Enumerating assignments is tiny for <=2 observations; unlike input
        # order or greedy nearest-neighbor, this handles result-order changes.
        best, best_cost = (None,) * len(detections), math.inf
        for assignment in itertools.product([None, *ids], repeat=len(detections)):
            used = [i for i in assignment if i is not None]
            if len(set(used)) != len(used):
                continue
            cost = 0.0
            for index, tid in enumerate(assignment):
                if tid is None:
                    cost += gate + 0.1
                    continue
                cost += costs[index,tid]
            if cost < best_cost:
                best, best_cost = assignment, cost
        hands = []
        hold = self.cfg["confirm_ms"] / 1000
        for d, tid, visible in zip(detections, best, visibility):
            visible = np.asarray(visible, dtype=bool)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                t = Track(tid, d.points.copy(), stamp)
                self.tracks[tid] = t
                self.event('track_created',tid)
            else:
                t = self.tracks[tid]
                dt = stamp - t.last
                if t.missing or dt > timeout + 1e-9:
                    t.handedness_state.reset_pending()
                    self.event('track_reacquired',tid,gap_s=dt,extended=dt>timeout+1e-9)
                tau = self.cfg["smoothing_ms"] / 1000
                alpha = 1.0 if tau == 0 else -math.expm1(-dt / tau)
                # A temporarily invisible track is reacquired without drawing
                # an interpolated skeleton across the occlusion gap.
                if t.missing or dt > 0.15:
                    alpha = 1.0
                    t.gesture = StableValue("Unknown")
                    t.pinch = StableValue(False)
                    t.released_at = -math.inf
                t.points = t.points + alpha * (d.points - t.points)
                t.last = stamp
                t.missing = False
            t.observed_points = d.points.copy()
            t.handedness = t.handedness_state.update(
                d.handedness, d.handedness_score, stamp, self.cfg)
            candidate = d.gesture if d.score >= self.cfg["gesture_confidence"] else "Unknown"
            if candidate == "Unknown":
                t.gesture = StableValue("Unknown")
            gesture = t.gesture.update(candidate, stamp, hold)
            score = d.score if candidate == gesture and gesture != "Unknown" else None
            ratio = pinch_ratio(d.points, visible, self.aspect)
            if ratio is None:
                t.pinch = StableValue(False)
                t.released_at = -math.inf
                state = "Unknown"
            else:
                before = t.pinch.value
                threshold = self.cfg["pinch_exit"] if before else self.cfg["pinch_enter"]
                closed = t.pinch.update(ratio <= threshold, stamp, hold)
                if before and not closed:
                    t.released_at = stamp
                state = ("Pinching" if closed else "Released" if
                         stamp - t.released_at < self.cfg["released_ms"] / 1000 else "Open")
            hands.append(Hand(tid, t.points.copy(), visible, t.handedness,
                              gesture, score, state, ratio,
                              t.handedness_state.evidence(d.handedness,d.handedness_score,stamp,self.cfg)))
            self.stats['observed'] += 1
            self.stats['unknown_side'] += t.handedness == 'Unknown'
            self.stats['side_'+hands[-1].identity['status']] += 1
        # Missing tracks survive briefly for association ONLY, never for display.
        observed_ids = {hand.id for hand in hands}
        for tid, track in self.tracks.items():
            if tid not in observed_ids and not track.missing:self.event('track_missing',tid)
            track.missing = tid not in observed_ids
        return hands
