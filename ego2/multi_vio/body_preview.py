"""Centered body models and timestamp-matched articulated hand preview.

Derived, bounded visualization only. It never supplies poses to inference or
recording. Per-device clocks are software anchored, not hardware synchronized.
"""
from collections import deque
from pathlib import Path
import bisect
import json
import math
import time
import numpy as np
import rerun as rr
import rerun.blueprint as b
from ..common.viewer import TIMELINE
from ..common.mahony import rotation, unit
from .hand_shape import HandSelector, HandShapeFilter, EDGES, template, PALM

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = dict(max_visual_age_s=.65, stale_after_s=.5, hide_after_s=1.5,
                max_clock_residual_s=.12, max_pose_gap_s=.15, roles={},
                hand_selection=dict(missing_grace=.25, switch_hold=.25, evidence_gap=.25, switch_margin=.10),
                hand_smoothing=dict(min_cutoff_hz=1.8, max_cutoff_hz=12., beta=.3,
                                    derivative_cutoff_hz=3., reset_gap_s=.25))


def load_settings():
    path = ROOT/'configs/body_preview.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    settings = {**DEFAULTS,**data}
    for name in ('max_visual_age_s','stale_after_s','hide_after_s','max_clock_residual_s','max_pose_gap_s'):
        value = float(settings[name])
        if not math.isfinite(value) or value <= 0: raise ValueError('Invalid body preview limit: '+name)
        settings[name] = value
    if settings['hide_after_s'] <= settings['stale_after_s']: raise ValueError('Invalid stale/hide limits')
    for group in ('hand_selection','hand_smoothing'):
        settings[group] = {**DEFAULTS[group],**data.get(group,{})}
        for key,value in settings[group].items():
            if key not in DEFAULTS[group] or not math.isfinite(float(value)) or float(value)<=0:
                raise ValueError('Invalid body preview setting: '+group+'.'+key)
            settings[group][key]=float(value)
    if settings['hand_smoothing']['max_cutoff_hz']<settings['hand_smoothing']['min_cutoff_hz']:
        raise ValueError('Invalid hand smoothing cutoff range')
    for role,values in settings['roles'].items():
        values['camera_from_body_wxyz'] = unit(values.get('camera_from_body_wxyz',[1,0,0,0])).tolist()
    return settings


def view(role, label):
    title = {'head':'Head attitude','chest':'Chest attitude','left_hand':'Left hand attitude','right_hand':'Right hand attitude'}[role]
    # Hands live in the initial camera frame: Y points through the palm, Z
    # follows the straight middle finger. View that XZ plane from the back
    # of the hand, with a small oblique angle to make its normal visible.
    hand = role in ('left_hand','right_hand')
    return b.Spatial3DView(name=title, origin=f'/body/{role}',
        contents='$origin/**', background=[10,10,10], line_grid=False,
        eye_controls=b.EyeControls3D(position=[.12,-.32,-.12] if hand else [.14,-.06,.32],
            look_target=[0,0,0],eye_up=[0,0,1] if hand else [0,-1,0]))


def slerp(a,b,f):
    a,b = unit(a),unit(b); dot = float(np.dot(a,b))
    if dot < 0: b=-b; dot=-dot
    if dot > .9995:return unit(a+f*(b-a))
    angle = math.acos(np.clip(dot,-1,1))
    return (math.sin((1-f)*angle)*a+math.sin(f*angle)*b)/math.sin(angle)


def gravity_display_frame(reference):
    """Initial cam0 -> fixed display with Y down and horizontal Z forward.

    Mahony's reference retains startup tilt; relative quaternions alone do
    not. Keep gravity vertical, choosing only unobservable heading at startup.
    For a lens pointing vertically, its horizontal right axis defines heading.
    """
    r = rotation(unit(reference))
    up = np.array([0.,0.,1.])
    forward = r[:,2] - up * r[2,2]
    if np.linalg.norm(forward) < .2:
        right = r[:,0] - up * r[2,0]
        right /= np.linalg.norm(right)
        forward = np.cross(up,right)
    forward /= np.linalg.norm(forward)
    down = -up
    right = np.cross(down,forward)
    return np.vstack((right,down,forward)) @ r


class PoseHistory:
    def __init__(self, max_gap=.15):
        self.rows = deque(maxlen=2400); self.max_gap=max_gap
        self.segment = 0; self.valid=False; self.reference_start=math.inf
        self.last = -math.inf

    def append(self, rows):
        for row in rows:
            stamp=float(row[0]); q=np.asarray(row[2:6],dtype=float)
            if not math.isfinite(stamp) or stamp <= self.last: continue
            valid = np.isfinite(q).all() and .95 < np.linalg.norm(q) < 1.05
            if valid and not self.valid:
                self.segment+=1; self.reference_start=stamp
            self.valid=bool(valid); self.last=stamp
            self.rows.append((stamp,unit(q) if valid else None,self.segment))
        while self.rows and self.last-self.rows[0][0] > 5.: self.rows.popleft()

    def at(self, stamp):
        if not self.valid or stamp < self.reference_start or not self.rows: return None
        rows=list(self.rows); i=bisect.bisect_left([r[0] for r in rows],stamp)
        if i<len(rows) and abs(rows[i][0]-stamp)<1e-7: return rows[i][1]
        if i==0 or i==len(rows): return None
        a,c=rows[i-1],rows[i]
        if a[1] is None or c[1] is None or a[2]!=c[2] or c[0]-a[0]>self.max_gap: return None
        return slerp(a[1],c[1],(stamp-a[0])/(c[0]-a[0]))


def part_geometry(role):
    if role in ('left_hand','right_hand'):
        p=template('Left' if role=='left_hand' else 'Right'); p-=p[PALM].mean(axis=0)
        return p,[p[[a,c]] for a,c in EDGES]
    if role=='head':
        a=np.linspace(0,2*np.pi,17); ring=np.column_stack((.055*np.sin(a),.072*np.cos(a)+.025,np.zeros(len(a))))
        nose=np.array([[0,.04,0],[0,.025,.033],[0,.012,0]])
        neck=np.array([[0,-.047,0],[0,-.095,0]])
        return np.concatenate((ring[:-1],nose[1:2],neck)),[ring,nose,neck]
    # Body-local +Y is up and +Z is the FRONT. Put the spine at the back
    # and a sternum at the front so this schematic isn't front/back symmetric.
    spine=np.array([[0,.09,-.025],[0,.04,-.025],[0,-.01,-.025],[0,-.06,-.025],[0,-.10,-.025]])
    shoulder=np.array([[-.085,.06,0],[-.06,.09,0],[.06,.09,0],[.085,.06,0]])
    lines=[spine,shoulder]
    for y,w in [(.045,.065),(.01,.075),(-.025,.07),(-.06,.055)]:
        a=np.linspace(0,2*np.pi,25); lines.append(np.column_stack((w*np.cos(a),np.full(len(a),y),.025*np.sin(a))))
    sternum=np.array([[0,.06,.036],[0,.025,.036],[0,-.025,.036],[0,-.065,.028]])
    lines.append(sternum)
    return np.concatenate((spine,shoulder,sternum)),lines


def front_marker(role):
    """A visible anatomical front landmark, in the same body-local frame."""
    if role=='head':return np.array([0,.025,.033]),'FACE'
    if role=='chest':return np.array([0,.025,.036]),'CHEST'
    raise ValueError('No rigid front marker for '+role)


class BodyPreview:
    def __init__(self, recording, roles, settings=None, diagnostics=None):
        self.recording=recording; self.roles=tuple(roles); self.settings=settings or load_settings()
        self.history={r:PoseHistory(self.settings['max_pose_gap_s']) for r in self.roles}
        self.clocks={}; self.clock_epochs={}; self.status={}; self.last_render={}; self.last_state={}
        self.state_stamps={}
        self.gravity_frames={}; self.gravity_keys={}; self.gravity_ready={}
        self.pending=deque(maxlen=24); self.diagnostics=diagnostics
        self.selector=HandSelector(max_age=self.settings['max_visual_age_s'],
            on_event=diagnostics.emit if diagnostics else None,
            **self.settings.get('hand_selection',DEFAULTS['hand_selection']))
        self.shape_filters={r:HandShapeFilter(**self.settings.get('hand_smoothing',DEFAULTS['hand_smoothing']))
                            for r in self.roles if 'hand' in r}
        self.pose_wait_keys={}; self.last_diagnostic=-math.inf
        self.tracking_stats={}
        self.mounts={r:rotation(unit(self.settings.get('roles',{}).get(r,{}).get('camera_from_body_wxyz',[1,0,0,0]))) for r in self.roles}
        self.stats=dict(visual_packets=0,clock_rejected=0,pose_wait=0,rendered=0,errors=0,
                        update_ms_total=0.,update_ms_max=0.)
        for role in self.roles:
            root=f'/body/{role}'
            recording.log(root,rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,static=True)
            # A static transparent extent keeps fitting independent of finger shape.
            recording.log(root+'/bounds',rr.Points3D([[-.15,-.15,-.15],[.15,.15,.15]],colors=[0,0,0,0],radii=0),static=True)
            p,lines=part_geometry(role)
            if role not in ('left_hand','right_hand'):
                # Temporal initial geometry: static colors would override later
                # tracking/stale colors at every timestamp in Rerun.
                self._time(0.)
                # Even the grey waiting model uses the mounting, so changing
                # to the first valid pose does not flip it upside down.
                recording.log(root+'/model',rr.Transform3D(translation=[0,0,0],mat3x3=self.mounts[role]))
                recording.log(root+'/model/points',rr.Points3D(p,colors=[110,110,110],radii=rr.Radius.ui_points(2.4)))
                recording.log(root+'/model/bones',rr.LineStrips3D(lines,colors=[110,110,110],radii=rr.Radius.ui_points(1.1)))
                self._front_marker(role,grey=True)
            self._state(role,0.,'WAITING: IMU initialization'+(' and visual hand' if 'hand' in role else ''))

    def _time(self,stamp):
        self.recording.reset_time(); self.recording.set_time(TIMELINE,duration=max(0.,float(stamp)))

    def _state(self,role,stamp,text):
        # A delayed capture can precede a wall-time stale status. Do not mark
        # it recovered in memory before the timeline can show that recovery.
        if stamp < self.state_stamps.get(role,-math.inf):return
        # Retain all repeated data timestamps but avoid flooding identical text.
        if text==self.last_state.get(role):return
        self._time(stamp);self.recording.log(f'/imu/{role}/orientation/body_status',rr.TextDocument(text))
        self.last_state[role]=text;self.state_stamps[role]=stamp

    def _mount_note(self,role):
        return '' if self.settings.get('roles',{}).get(role,{}).get('mount_calibrated',False) else ' | reference axes; mount uncalibrated'

    def consume_imu(self,packet,now):
        role=packet['role']; history=self.history[role]
        clock=packet.get('clock',{})
        offset=clock.get('clock_offset')
        if offset is not None and math.isfinite(float(offset)):
            epoch=clock.get('clock_epoch',0)
            if role in self.clock_epochs and epoch!=self.clock_epochs[role]:
                history=PoseHistory(self.settings['max_pose_gap_s']);self.history[role]=history
                self.gravity_frames.pop(role,None);self.gravity_keys.pop(role,None)
                self.selector.reset_role(role,now)
                self.pending=deque((p for p in self.pending if p['role']!=role),maxlen=24)
            self.clocks[role]=float(offset);self.clock_epochs[role]=epoch
        history.append(packet['imu'])
        if packet.get('status'):self.status[role]=packet['status'][-1][1]
        if role not in ('left_hand','right_hand') and history.valid and history.rows:
            stamp,q,_=history.rows[-1]
            ref=packet.get('body_reference')
            self.gravity_ready[role]=bool(ref and math.isfinite(float(ref['start'])) and ref['start']<=stamp)
            if self.gravity_ready[role]:
                key=(self.clock_epochs.get(role,0),ref['segment'],ref['start'])
                if self.gravity_keys.get(role)!=key:
                    self.gravity_frames[role]=gravity_display_frame(ref['world_from_initial_camera_wxyz'])
                    self.gravity_keys[role]=key
                if now-stamp<=self.settings['max_visual_age_s']:
                    self._draw(role,stamp,q,None,'IMU: '+role+' | gravity aligned; heading relative | '+self.status.get(role,'TRACKING'))
        self._drain_pending(now)

    def consume_visual(self,packet,now):
        self.stats['visual_packets']+=1
        if len(self.pending)==self.pending.maxlen:
            old=self.pending[0]
            self.selector.event('packet_filter','pending_overflow',(old['role'],),now)
        self.pending.append(packet);self._drain_pending(now)

    def _drain_pending(self,now):
        waiting=deque(maxlen=24)
        while self.pending:
            p=self.pending.popleft();role=p['role']
            if now-float(p['received'])>self.settings['max_visual_age_s']:
                self.selector.event('packet_filter','visual_too_old',(role,),now);continue
            if role not in self.clocks:waiting.append(p);continue
            device=p.get('image_device_time')
            if device is None or not math.isfinite(float(device)):
                self.selector.event('packet_filter','missing_device_time',(role,),now);continue
            stamp=float(device)+self.clocks[role]
            residual=float(p['received'])-stamp
            if abs(residual)>self.settings['max_clock_residual_s'] or stamp>now+.02:
                self.stats['clock_rejected']+=1
                self.selector.event('packet_filter','clock_residual',(role,),stamp,residual_s=residual);continue
            if stamp > self.selector.last_stamps.get(role,-math.inf):
                for eye,row in p.get('tracking',{}).items():
                    self.tracking_stats[role+'/'+eye]=dict(row['stats'])
                    if self.diagnostics:
                        for event in row['events']:
                            self.diagnostics.emit(dict(event, kind='tracking', role=role, eye=eye, stamp=stamp))
            self.selector.update(role,stamp,p['observations'])
        self.pending=waiting

    def _draw(self,role,stamp,q,shape,source):
        previous=self.last_render.get(role,{})
        if stamp<=previous.get('stamp',-math.inf):return
        c=self.mounts[role];camera_rotation=rotation(q)
        if role in ('head','chest') and role in self.gravity_frames:
            camera_rotation=self.gravity_frames[role]@camera_rotation
        # q maps current camera coordinates to initial camera coordinates.
        # Keep every model in that SAME frame as its camera axes: R_cam @ C.
        # C.T @ R_cam @ C instead renders in initial body coordinates; at
        # identity it cancels the mounting (including the head/chest Y flip).
        r=camera_rotation@c
        root=f'/body/{role}/model'
        self._time(stamp)
        self.recording.log(root,rr.Transform3D(translation=[0,0,0],mat3x3=r))
        self.recording.log(f'/body/{role}/camera',rr.Transform3D(translation=[0,0,0],mat3x3=camera_rotation))
        self._camera_axes(role)
        if shape is not None:
            color=[75,210,255] if role=='left_hand' else [255,180,60]
            self.recording.log(root+'/points',rr.Points3D(shape,colors=color,radii=rr.Radius.ui_points(2.4)))
            self.recording.log(root+'/bones',rr.LineStrips3D([shape[[a,c]] for a,c in EDGES],colors=color,radii=rr.Radius.ui_points(1.1)))
        elif role not in self.last_render or self.last_render[role]['grey'] or self.last_render[role]['hidden']:
            p,lines=part_geometry(role)
            self.recording.log(root+'/points',rr.Points3D(p,colors=[105,215,235],radii=rr.Radius.ui_points(2.4)))
            self.recording.log(root+'/bones',rr.LineStrips3D(lines,colors=[105,215,235],radii=rr.Radius.ui_points(1.1)))
            if role in ('head','chest'):self._front_marker(role)
        # Gray/Clear are timestamped when loss is detected. A resumed packet
        # may still be older than that marker. Keep recovery pending and emit
        # full geometry until a capture passes it; otherwise a rigid model's
        # colors/positions can remain gray/cleared forever while it rotates.
        appearance_stamp=previous.get('appearance_stamp',0.)
        covered=stamp<=appearance_stamp
        self.last_render[role]=dict(stamp=stamp,shape=shape,
            grey=covered and previous.get('grey',False),
            hidden=covered and previous.get('hidden',False),
            appearance_stamp=appearance_stamp)
        self.stats['rendered']+=1
        self._state(role,stamp,source+self._mount_note(role))

    def _camera_axes(self,role,grey=False):
        # Dynamic camera axes share the skeleton's timestamp and are hidden
        # with it on tracking loss. No independent/latest-pose overlay.
        labels={'head':['X+','Y+ down','Z+ face'],
                'chest':['X+','Y+ down','Z+ chest']}.get(role,['X+','Y+ palm','Z+ forward'])
        self.recording.log(f'/body/{role}/camera/axes',rr.Arrows3D(
            origins=[[0,0,0]]*3,vectors=np.eye(3)*.045,
            colors=[[110,110,110]]*3 if grey else [[239,95,98],[89,204,124],[86,155,255]],
            radii=.0005,labels=labels,show_labels=True))

    def _front_marker(self,role,grey=False):
        position,label=front_marker(role)
        self.recording.log(f'/body/{role}/model/front',rr.Points3D([position],
            colors=[110,110,110] if grey else [255,190,70],
            radii=rr.Radius.ui_points(3.5),labels=[label],show_labels=True))

    def tick(self,now):
        started=time.perf_counter()
        try:
            for role,side in (('left_hand','Left'),('right_hand','Right')):
                if role not in self.history:continue
                c=self.selector.choose(side,now)
                if c is not None:
                    q=self.history[role].at(c.stamp)
                    if q is not None:
                        if c.stamp>self.last_render.get(role,{}).get('stamp',-math.inf):
                            epoch=(self.history[role].segment,self.clock_epochs.get(role,0),
                                   self.clock_epochs.get(c.source[0],0))
                            shape=self.shape_filters[role].update(c.shape,side,c.stamp,c.source,epoch)
                            src='/'.join(map(str,c.source))
                            self._draw(role,c.stamp,q,shape,f'IMU: {role} | hand: {src} | age {max(0.,now-c.stamp)*1000:.0f} ms')
                    else:
                        self.stats['pose_wait']+=1
                        key=(c.source,c.stamp)
                        if self.pose_wait_keys.get(role)!=key:
                            self.pose_wait_keys[role]=key
                            self.selector.event('pose_wait','matching_pose_unavailable',c.source,c.stamp,side=side)
            for role in self.roles:
                last=self.last_render.get(role);history=self.history[role]
                reason=None
                if not history.valid:reason=self.status.get(role,'WAITING: IMU initialization')
                elif role in ('head','chest') and not self.gravity_ready.get(role,False):reason='WAITING: gravity reference'
                elif now-history.last>self.settings['stale_after_s']:reason='IMU STALE'
                elif last is None or now-last['stamp']>(
                        max(self.settings['stale_after_s'],self.settings['max_visual_age_s'])
                        if 'hand' in role else self.settings['stale_after_s']):
                    reason='WAITING: visible, confirmed hand and matching IMU' if 'hand' in role else 'IMU STALE'
                if reason:
                    if last and not last['hidden']:
                        self._time(now);root=f'/body/{role}/model'
                        if now-last['stamp']>self.settings['hide_after_s']:
                            self.recording.log(root,rr.Clear(recursive=True));last['hidden']=True
                            self.recording.log(f'/body/{role}/camera',rr.Clear(recursive=True))
                            last['appearance_stamp']=now
                        elif not last['grey']:
                            # Only color is a current-time status. Repeating old
                            # joints at now would override a later-arriving
                            # capture's geometry but not its paired transform.
                            self.recording.log(root+'/points',rr.Points3D.from_fields(colors=[110,110,110]))
                            self.recording.log(root+'/bones',rr.LineStrips3D.from_fields(colors=[110,110,110]));last['grey']=True
                            last['appearance_stamp']=now
                            self._camera_axes(role,grey=True)
                            if role in ('head','chest'):self._front_marker(role,grey=True)
                    self._state(role,now,reason+self._mount_note(role))
            if self.diagnostics and now-self.last_diagnostic>=5.:
                self.last_diagnostic=now
                self.diagnostics.emit(dict(kind='summary',stamp=now,**self.summary()))
        finally:
            elapsed=(time.perf_counter()-started)*1000
            self.stats['update_ms_total']+=elapsed;self.stats['update_ms_max']=max(self.stats['update_ms_max'],elapsed)

    def summary(self):
        return dict(selector=dict(self.selector.stats), reasons=dict(self.selector.reasons),
                    tracking={k:dict(v) for k,v in self.tracking_stats.items()},
                    states=dict(self.last_state),
                    models={r:{k:v[k] for k in ('stamp','grey','hidden','appearance_stamp')}
                            for r,v in self.last_render.items()},
                    accepted_sides=dict(self.selector.accepted_sides),
                    sources={k:dict(v) for k,v in self.selector.sources.items()},
                    diagnostics=dict(getattr(self.diagnostics,'stats',{})),
                    selected={k:list(v) for k,v in self.selector.selected.items()},
                    smoothing={k:dict(v.stats) for k,v in self.shape_filters.items()})

    def close(self):
        if self.diagnostics: self.diagnostics.close(self.summary())
