"""Palm-local, fixed-length visual hand shapes. No camera/world triangulation.

All operations are on existing recognizer results, never on capture images.
Model world landmarks are estimates; handedness score is not landmark confidence.
Hand-local axes: +Y wrist to middle finger, +Z out of the back of the hand
(palm normal is -Z). Mirroring Left in X preserves this convention for both sides.
"""
from collections import Counter
from dataclasses import dataclass
import math
import numpy as np
from ..gesture.tracking import usable_side_evidence

PALM = [0, 5, 9, 13, 17]
CHAINS = [(0,1,2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
         (13,17),(0,17),(17,18),(18,19),(19,20)]


def template(side):
    p = np.zeros((21,3), dtype=float)
    p[1:5] = [[.032,.025,0],[.053,.043,0],[.068,.060,0],[.080,.075,0]]
    for root,x,y,lengths in [(5,.026,.067,[.039,.025,.020]),
                             (9,0,.074,[.043,.028,.022]),
                             (13,-.023,.068,[.040,.026,.020]),
                             (17,-.043,.055,[.030,.020,.018])]:
        p[root] = [x,y,0]
        for i,length in enumerate(lengths,1): p[root+i] = p[root+i-1]+[0,length,0]
    if side == 'Left': p[:,0] *= -1
    elif side != 'Right': raise ValueError('Unknown anatomical hand side')
    return p


def palm_local(points, side):
    p = np.asarray(points, dtype=float)
    if p.shape != (21,3) or not np.isfinite(p).all(): raise ValueError('Invalid world landmarks')
    y = p[9]-p[0]; size = np.linalg.norm(y)
    if not .015 < size < .25: raise ValueError('Invalid estimated palm size')
    y /= size
    x = p[5]-p[17]; x -= y*np.dot(x,y)
    width = np.linalg.norm(x)
    if not .2 < width/size < 2.: raise ValueError('Degenerate palm basis')
    x /= width
    if side == 'Left': x = -x
    elif side != 'Right': raise ValueError('Unconfirmed side')
    basis = np.column_stack((x,y,np.cross(x,y)))
    return (p-p[0])@basis/size


def retarget(local, side):
    """Preserve observed bone directions, use stable schematic bone lengths."""
    target = template(side); out = target.copy()
    scale = np.linalg.norm(target[9]-target[0])
    for chain in CHAINS:
        for a,b in zip(chain,chain[1:]):
            v = local[b]-local[a]; length = np.linalg.norm(v)
            reference = np.linalg.norm(target[b]-target[a])
            if not .18 < length/(reference/scale) < 3.5: raise ValueError('Implausible bone length')
            out[b] = out[a]+v*(reference/length)
    # Fixed palm origin, not a bounding-box centroid that moves when fingers bend.
    return out-target[PALM].mean(axis=0)


def pack_observations(observations, hands, eye_names):
    """Join stable tracks to raw 3D points without changing stored detections."""
    packed = []
    for eye,(obs,tracks) in enumerate(zip(observations,hands)):
        by_id = {h.id:h for h in tracks}
        for item in obs:
            hand = by_id.get(item.get('track_id'))
            identity = getattr(hand, 'identity', None)
            reason = None
            if hand is None: reason = 'no_accepted_track'
            elif hand.handedness not in ('Left','Right'): reason = 'side_unconfirmed'
            elif identity is not None:
                if not usable_side_evidence(identity, hand.handedness):
                    reason = {'conflict':'side_conflict', 'expired':'side_evidence_expired',
                              'unconfirmed':'side_unconfirmed'}.get(
                                  identity.get('status') if isinstance(identity, dict) else None, 'invalid_side_evidence')
            elif item.get('handedness') != hand.handedness: reason = 'side_conflict'
            if reason is None and hand is not None and item.get('world_points') is None:
                reason = 'no_world_landmarks'
            if reason:
                # Carry rejection metadata only, never substitute model points.
                packed.append(dict(eye=eye_names[eye], track_id=item.get('track_id'),
                    side=getattr(hand, 'handedness', 'Unknown'), rejection_reason=reason))
                continue
            world = item['world_points']
            packed.append(dict(eye=eye_names[eye], track_id=int(hand.id), side=hand.handedness,
                identity=dict(identity) if identity is not None else None,
                raw_handedness=item.get('handedness'),
                handedness_score=float(item.get('handedness_score',0)),
                points=np.asarray(item['points'],dtype=np.float32).copy(),
                world_points=np.asarray(world,dtype=np.float32).copy(),
                visible=np.asarray(item.get('visible',[]),dtype=bool).copy()))
    return packed


@dataclass(eq=False)
class Candidate:
    source: tuple
    side: str
    stamp: float
    shape: np.ndarray
    quality: float


def candidate(role, stamp, item):
    side = item['side']; score = float(item['handedness_score'])
    identity = item.get('identity')
    if identity is None:
        # Old recordings have no temporal evidence; keep their strict gate.
        if side not in ('Left','Right') or not .75 <= score <= 1.: raise ValueError('Uncertain side')
    elif not usable_side_evidence(identity, side) or not 0 <= score <= 1.:
        raise ValueError('Invalid side evidence')
    xy = np.asarray(item['points'],dtype=float)
    visible = np.asarray(item['visible'],dtype=bool)
    if xy.shape != (21,3) or visible.shape != (21,) or not np.isfinite(xy).all():
        raise ValueError('Invalid image landmarks')
    inside = visible & (xy[:,:2] >= .015).all(axis=1) & (xy[:,:2] <= .985).all(axis=1)
    if not inside[PALM].all() or inside.sum() < 18: raise ValueError('Cropped hand')
    span = np.ptp(xy[:,:2],axis=0)
    if min(span) < .025 or max(span) < .08: raise ValueError('Hand too small')
    local = palm_local(item['world_points'],side)
    shape = retarget(local,side)
    quality = .35*score+.35*float(inside.mean())+.30*min(1.,float(np.sqrt(np.prod(span)))/.25)
    return Candidate((role,item['eye'],int(item['track_id'])),side,float(stamp),shape,quality)


REJECTION_CODES = {
    'Invalid side evidence': 'invalid_side_evidence',
    'Uncertain side': 'side_score_low', 'Invalid image landmarks': 'invalid_image_points',
    'Cropped hand': 'cropped_hand', 'Hand too small': 'hand_too_small',
    'Invalid world landmarks': 'invalid_world_points', 'Invalid estimated palm size': 'palm_size',
    'Degenerate palm basis': 'palm_basis', 'Unconfirmed side': 'side_unconfirmed',
    'Implausible bone length': 'bone_length',
}


class HandSelector:
    """Whole-hand selection; a lost source gets a short grace, never stale output."""
    def __init__(self, max_age=.65, switch_margin=.10, switch_hold=.25,
                 missing_grace=.25, evidence_gap=.25, on_event=None):
        self.candidates = {}; self.selected = {}; self.challengers = {}
        self.missing_since = {}; self.last_stamps = {}; self.output_stamps = {}
        self.max_age,self.switch_margin,self.switch_hold = max_age,switch_margin,switch_hold
        self.missing_grace,self.evidence_gap = missing_grace,evidence_gap
        self.on_event = on_event
        self.stats = dict(accepted=0,rejected=0,switches=0,conflicts=0)
        self.reasons = Counter(); self.sources = {}; self.accepted_sides = Counter()

    def event(self, kind, reason, source, stamp, **fields):
        self.reasons[reason] += 1
        if self.on_event:
            self.on_event(dict(kind=kind, reason=reason, source=list(source), stamp=float(stamp) if math.isfinite(stamp) else None, **fields))

    def reject(self, reason, source, stamp, conflict=False):
        self.stats['rejected'] += 1
        if conflict: self.stats['conflicts'] += 1
        key = '/'.join(map(str,source[:2]))
        self.sources.setdefault(key,dict(accepted=0,rejected=0))['rejected'] += 1
        self.event('filter',reason,source,stamp)

    def reset_role(self, role, stamp):
        self.candidates = {k:v for k,v in self.candidates.items() if k[0] != role}
        self.last_stamps.pop(role,None)
        self.challengers.clear()
        self.event('reset','clock_epoch_reset',(role,),stamp)

    def update(self, role, stamp, observations):
        if not math.isfinite(stamp) or stamp <= self.last_stamps.get(role,-math.inf):
            self.event('packet_filter','nonmonotonic_visual',(role,),stamp)
            return
        self.last_stamps[role] = stamp
        old = {k:v for k,v in self.candidates.items() if k[0] == role}
        self.candidates = {k:v for k,v in self.candidates.items() if k[0] != role}
        valid = []
        if not observations: self.event('observation','no_hand_observation',(role,),stamp)
        for item in observations:
            source = (role,item.get('eye','unknown'),item.get('track_id'))
            reason = item.get('rejection_reason')
            if reason:
                self.reject(reason,source,stamp,reason=='side_conflict'); continue
            try: c = candidate(role,stamp,item)
            except (ValueError,KeyError,TypeError) as exc:
                self.reject(REJECTION_CODES.get(str(exc),'malformed_observation'),source,stamp); continue
            previous = old.get(c.source)
            if previous and previous.side != c.side:
                self.reject('track_side_changed',source,stamp,True); continue
            if previous:
                change = float(np.mean(np.linalg.norm(c.shape-previous.shape,axis=1)))/.074
                if change > .9:
                    self.reject('shape_jump',source,stamp); continue
                c.quality -= min(.15,change*.15)
            valid.append(c)
        for c in valid:
            if sum(v.source[:2] == c.source[:2] and v.side == c.side for v in valid) != 1:
                self.reject('ambiguous_same_side',c.source,stamp,True); continue
            self.candidates[c.source] = c; self.stats['accepted'] += 1
            self.accepted_sides[c.side] += 1
            key = '/'.join(map(str,c.source[:2]))
            self.sources.setdefault(key,dict(accepted=0,rejected=0))['accepted'] += 1
        # One completed frame invalidates its missing hand; retain only source
        # preference during grace, not the previous observation as new data.
        for side,source in self.selected.items():
            if source[0] == role and source not in self.candidates:
                if side not in self.missing_since:
                    self.missing_since[side] = stamp
                    self.event('loss','selected_source_missing',source,stamp,side=side)
            elif source[0] == role:
                self.missing_since.pop(side,None)
        for side,entry in list(self.challengers.items()):
            if entry['source'][0] == role and entry['source'] not in self.candidates:
                self.challengers.pop(side,None)

    def _confirmed(self, side, c, reason):
        entry = self.challengers.get(side)
        if (entry is None or entry['source'] != c.source or entry['reason'] != reason
                or c.stamp-entry['last'] > self.evidence_gap):
            entry = dict(source=c.source, reason=reason, first=c.stamp, last=c.stamp, samples=1)
            self.challengers[side] = entry
        elif c.stamp > entry['last']:
            entry['last']=c.stamp; entry['samples']+=1
        # Repeated viewer ticks on ONE observation cannot confirm a source.
        return entry['samples'] >= 2 and c.stamp-entry['first'] >= self.switch_hold-1e-9

    def choose(self, side, now):
        floor = self.output_stamps.get(side,-math.inf)
        pool = [c for c in self.candidates.values() if c.side == side
                and 0 <= now-c.stamp <= self.max_age and c.stamp >= floor]
        if not pool:
            self.challengers.pop(side,None); return None
        rank = lambda c:c.quality-.3*(now-c.stamp)
        best = max(pool,key=rank)
        source = self.selected.get(side)
        current = self.candidates.get(source)
        if current not in pool: current = None
        chosen = current; reason = 'initial_source'
        if source is None:
            chosen = best
        elif current is None:
            lost = self.missing_since.setdefault(side,now)
            # A replacement must advance capture time. A higher-scoring
            # previously emitted sample cannot block a genuinely new one.
            replacements = [c for c in pool if c.stamp > floor]
            if not replacements:
                self.challengers.pop(side,None); return None
            best = max(replacements,key=rank)
            # Asynchronous cameras alternate which sample is newest. Keep a
            # healthy challenger through small score changes so confirmation
            # cannot starve forever while multiple good cameras keep updating.
            entry = self.challengers.get(side)
            preferred = self.candidates.get(entry['source']) if entry else None
            if (preferred in replacements
                    and rank(best)-rank(preferred) <= self.switch_margin):
                best = preferred
            ready = self._confirmed(side,best,'source_lost')
            # Hysteresis protects a briefly missing source, but must not pin a
            # dead track forever. After the normal observation lifetime, treat
            # a NEW fresh, fully validated observation as reacquisition. Track
            # churn or intermittent filtering cannot extend this deadline.
            expired = now-lost >= self.max_age
            # max_age already bounds capture-to-display latency in the pool.
            # evidence_gap bounds spacing of distinct capture timestamps in
            # _confirmed, NOT inference latency (four cameras often >250 ms).
            if expired:
                chosen = best; reason = 'source_reacquired'
            elif now-lost >= self.missing_grace and ready:
                chosen = best; reason = 'source_lost'
        elif best.source == current.source:
            self.challengers.pop(side,None); self.missing_since.pop(side,None)
        else:
            quality = abs(best.stamp-current.stamp) <= .08 and rank(best)>rank(current)+self.switch_margin
            stale = now-current.stamp > .4 and best.stamp > current.stamp+.1
            reason = 'quality_advantage' if quality else 'source_stale'
            if (quality or stale) and self._confirmed(side,best,reason): chosen=best
            elif not (quality or stale): self.challengers.pop(side,None)
        if chosen is None: return None
        if chosen.source != source:
            if source is not None: self.stats['switches'] += 1
            self.selected[side] = chosen.source
            self.challengers.pop(side,None); self.missing_since.pop(side,None)
            self.event('source',reason,chosen.source,chosen.stamp,side=side,
                       previous=list(source) if source is not None else None, switches=self.stats['switches'],
                       quality=float(chosen.quality), age_s=float(now-chosen.stamp))
        self.output_stamps[side] = chosen.stamp
        return chosen


class HandShapeFilter:
    """Causal adaptive bone-direction filter, at the selected observation time.

    No frame queue, prediction, IMU filtering or cross-source blending. Rebuild
    from fixed bone lengths so smoothing cannot shorten fingers or move the palm.
    """
    def __init__(self, min_cutoff_hz=1.8, max_cutoff_hz=12., beta=.3,
                 derivative_cutoff_hz=3., reset_gap_s=.25):
        self.low,self.high,self.beta = min_cutoff_hz,max_cutoff_hz,beta
        self.derivative,self.gap = derivative_cutoff_hz,reset_gap_s
        self.state=None; self.stats=dict(updates=0,resets=0,alpha_mean=1.)

    def reset(self): self.state=None

    @staticmethod
    def alpha(cutoff, dt): return -np.expm1(-2*np.pi*cutoff*dt)

    def update(self, shape, side, stamp, source, epoch=0):
        p=np.asarray(shape,dtype=float)
        edges=[(a,b) for chain in CHAINS for a,b in zip(chain,chain[1:])]
        vectors=np.array([p[b]-p[a] for a,b in edges])
        lengths=np.linalg.norm(vectors,axis=1)
        if not np.isfinite(p).all() or np.any(lengths<1e-8): raise ValueError('Invalid filter input')
        raw=vectors/lengths[:,None]; key=(side,source,epoch)
        old=self.state; dt=stamp-old['stamp'] if old else 0.
        if old and key==old['key'] and dt<=0: raise ValueError('Nonmonotonic filter time')
        if old is None or old['key']!=key or dt>self.gap or dt<=0:
            directions=raw.copy(); speed=np.zeros(len(edges)); alpha=np.ones(len(edges))
            self.stats['resets']+=1
        else:
            angular=np.arccos(np.clip(np.sum(raw*old['raw'],axis=1),-1,1))/dt
            speed=old['speed']+self.alpha(self.derivative,dt)*(angular-old['speed'])
            cutoff=np.clip(self.low+self.beta*speed,self.low,self.high)
            alpha=self.alpha(cutoff,dt)
            directions=(1-alpha[:,None])*old['directions']+alpha[:,None]*raw
            norm=np.linalg.norm(directions,axis=1)
            bad=norm<1e-6
            directions[bad]=raw[bad]; norm[bad]=1.
            directions/=norm[:,None]
        out=template(side); reference=out.copy()
        for (a,b),direction in zip(edges,directions):
            out[b]=out[a]+direction*np.linalg.norm(reference[b]-reference[a])
        out-=reference[PALM].mean(axis=0)
        self.state=dict(stamp=stamp,key=key,raw=raw.copy(),directions=directions.copy(),speed=speed)
        self.stats['updates']+=1;self.stats['alpha_mean']=float(alpha.mean())
        return out
