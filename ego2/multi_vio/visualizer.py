"""One official Rerun Viewer for all configured stereo-inertial cameras."""
from collections import deque
import copy
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as b
from ..common.viewer import ViewerConnection, TIMELINE
from .config import roles_for, camera_label
from .roles import reference_for

COLORS = {'chest':[80,160,255],
          'left_hand':[255,165,70], 'right_hand':[63,220,190], 'head':[193,138,255]}
AXES = [[255,65,65],[65,230,100],[65,140,255]]
NAMES = {'chest':'Chest','head':'Head','left_hand':'Left hand','right_hand':'Right hand'}
RELATIVE_ORDER = ('left_hand','right_hand','head')


def draw_compact_id(image,pixels,label,color):
    """Scale ID text with the image instead of fixed-size viewport labels.

    Caller supplies a display copy. The detector's RGB and pose pixels stay
    untouched; full rejection reasons remain on the Rerun point annotation.
    """
    side=float(np.min(np.linalg.norm(np.roll(pixels,-1,axis=0)-pixels,axis=1)))
    text=label.split()[0]
    scale=float(np.clip(side*.22/22,.25,.9))
    thickness=max(1,round(scale*2))
    (width,height),baseline=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
    x=int(np.clip(pixels[:,0].mean()-width/2,0,max(0,image.shape[1]-width-2)))
    y=int(np.clip(pixels[:,1].min()-3,height+2,image.shape[0]-baseline-2))
    cv2.rectangle(image,(x-1,y-height-2),(x+width+1,y+baseline+1),(28,28,28),-1)
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,tuple(color[::-1]),thickness,cv2.LINE_AA)


def initial_multi_eye_geometry(poses,viewer):
    """Look along initial reference optical +Z with gravity-up on screen."""
    reference=poses[reference_for(poses)]
    forward=reference[:3,2]
    positions=np.array([pose[:3,3] for pose in poses.values()])
    target=positions.mean(axis=0)
    radius=np.linalg.norm(positions-target,axis=1).max()+viewer['axis_length_m']
    distance=max(viewer['view_distance_m'],3*radius)
    up=np.array([0.,0.,1.])
    up-=forward*np.dot(up,forward)
    # Looking straight up/down has no gravity-defined screen-up direction.
    # Use optical -Y there, while keeping the shared world itself Z-up.
    if np.linalg.norm(up)<1e-4:
        up=-reference[:3,1]
    up=up/np.linalg.norm(up)
    return dict(position=target-distance*forward,look_target=target,eye_up=up,spin_speed=0.)


def build_blueprint(cfg):
    roles=roles_for(cfg)
    rows = [b.Horizontal(*[b.Spatial2DView(name=f'{NAMES[role]} {camera_label(cfg,role)} | {"Left" if index==0 else "Right"} / {cam}',
                    origin=f'/cameras/{role}/{cam}',contents=f'/cameras/{role}/{cam}/**')
                    for index,cam in enumerate(cfg['viewer']['display_order'])],column_shares=[1,1])
            for role in cfg['viewer'].get('row_order',roles)]
    left=b.Spatial3DView(name='Camera poses and trajectories (Z up)',origin='/world',contents='/world/**',
                       line_grid=False,background=b.Background(color=[32,32,32]))
    if cfg['viewer'].get('initial_pose_panel'):
        share=cfg['viewer']['initial_pose_share']
        left=b.Vertical(left,b.TextDocumentView(name='Initial relative pose [Δp/m, ΔR/deg]',
            origin='/initial_relative_pose',contents='$origin/readout'),row_shares=[1-share,share])
    return b.Blueprint(b.Horizontal(left,
        b.Vertical(*rows,row_shares=[1]*len(rows)),column_shares=[.55,.45]),
        b.TimePanel(state='expanded',timeline=TIMELINE,play_state='following'),
        b.BlueprintPanel(state='hidden'),b.SelectionPanel(state='hidden'),auto_layout=False,auto_views=False)


class Visualizer(ViewerConnection):
    def __init__(self,cfg,devices,args,save=None):
        self.cfg,self.devices=cfg,devices
        self.roles=tuple(devices)
        self.recording=rr.RecordingStream('ego2-'+cfg['mode'].replace('_','-')+('-demo' if args.demo else ''))
        self.blueprint=build_blueprint(cfg)
        self.eye_initialized=False
        self.web_process=None
        self.display_processors={}
        self.live_preview=None
        self.board_status={r:'Board: waiting' for r in self.roles}
        self.last_valid={r:False for r in self.roles}
        self.paths={r:dict(epoch=-1,segment=0,chunk=0,points=[],entities=deque(),last=None) for r in self.roles}
        try:
            self._configure(cfg,args.headless,save,args.connect,args.web,args.web_port,args.grpc_port)
            self.recording.log('/world',rr.ViewCoordinates.RIGHT_HAND_Z_UP,static=True)
            self.time(0.)
            self.initial_relative(None)
            from ..common.display_isp import create_display_isp
            for role in self.roles:
                self.display_processors[role] = create_display_isp(cfg, devices[role]['native']['serial'], args)
            if not args.demo:
                from .live_preview import TrajectoryPreview
                self.live_preview=TrajectoryPreview(self.roles,self.display_processors,
                    cfg['viewer']['jpeg_quality'],cfg['viewer'].get('display_isp',{}).get('max_fps',15),
                    self._preview_images,self._preview_overlays)
                print('[vio-preview] 画面不等待标定板检测；隐藏检测框，顶部状态合并显示。',flush=True)
        except BaseException:
            self.close(); raise

    def time(self,stamp):
        self.recording.set_time(TIMELINE,duration=float(stamp))

    def preview(self,role,packet,images,status,*,raw_transport=None):
        self.live_preview.submit(role,packet,images,status,raw_transport)

    def board_overlay(self,role,packet,observations,track,status,processing_ms):
        # Board detection still feeds alignment and recording in Output.board.
        # Live preview only keeps its latest status, never late frame geometry.
        self._preview_overlays(role,packet,observations,track,status,processing_ms)

    def _status_text(self,role,status):
        board=getattr(self,'board_status',{}).get(role,'Board: waiting')
        return f'{status}\n{board}'

    def _preview_images(self,role,packet,encoded,status):
        self.time(packet['capture_time'])
        text=self._status_text(role,status)
        for cam,image in zip(('cam0','cam1'),encoded):
            root=f'/cameras/{role}/{cam}'
            # Also clear the old separate board-status label below this path.
            self.recording.log(root+'/overlays',rr.Clear(recursive=True))
            self.recording.log(root+'/image',image)
            # One multiline label keeps UI line spacing at every image scale.
            self.recording.log(root+'/status',rr.Points2D([[640,35]],labels=[text],
                show_labels=True,colors=[235,235,235],radii=rr.Radius.ui_points(1)))

    def _preview_overlays(self,role,packet,observations,track,status,processing_ms):
        if not hasattr(self,'board_status'):
            self.board_status={}
        self.board_status[role]=f'Board: {track.status} | {processing_ms:.0f} ms'

    def images(self,role,packet,images,observations,track,status,processing_ms,*,raw_transport=None):
        processor = getattr(self, 'display_processors', {}).get(role)
        if processor:
            packet, observations, track = copy.deepcopy((packet, observations, track))
            processor.submit(raw_transport, images, lambda shown:
                self._images(role,packet,shown,observations,track,status,processing_ms),
                require_enhanced=True)
        else:
            self._images(role,packet,images,observations,track,status,processing_ms)

    def _images(self,role,packet,images,observations,track,status,processing_ms):
        self._preview_overlays(role,packet,observations,track,status,processing_ms)
        self.time(packet['capture_time'])
        rec=self.recording
        text=self._status_text(role,status)
        for cam,image in zip(('cam0','cam1'),images):
            root=f'/cameras/{role}/{cam}'
            rec.log(root+'/overlays',rr.Clear(recursive=True))
            rec.log(root+'/image',rr.Image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).compress(jpeg_quality=self.cfg['viewer']['jpeg_quality']))
            rec.log(root+'/status',rr.Points2D([[image.shape[1]/2,35]],labels=[text],show_labels=True,colors=[235,235,235],radii=rr.Radius.ui_points(1)))

    def close(self):
        if getattr(self,'live_preview',None):
            self.live_preview.close()
            self.live_preview=None
            self.display_processors={}
        else:
            for processor in getattr(self, 'display_processors', {}).values():
                if processor: processor.close()
        super().close()

    def state(self,role,stamp,status):
        self.time(stamp)
        text=self._status_text(role,status)
        for cam in ('cam0','cam1'):
            self.recording.log(f'/cameras/{role}/{cam}/status',rr.Points2D([[640,35]],labels=[text],show_labels=True,colors=[255,190,60],radii=rr.Radius.ui_points(1)))

    def initialize_poses(self,poses):
        if self.eye_initialized or set(poses)!=set(roles_for(self.cfg)) or any(p is None for p in poses.values()):
            return
        # Retain the existing view IDs. Only seed the view once, so later
        # motion/map resets cannot override the user's orbit or zoom.
        view=self.blueprint.root_container.contents[0]
        if self.cfg['viewer'].get('initial_pose_panel'):
            view=view.contents[0]
        view.properties['EyeControls3D']=b.EyeControls3D(**initial_multi_eye_geometry(poses,self.cfg['viewer']))
        self.recording.send_blueprint(self.blueprint)
        self.eye_initialized=True

    def initial_relative(self,event):
        if not self.cfg['viewer'].get('initial_pose_panel'):
            return
        if event is not None:
            self.time(event['time'])
        self.recording.log('/initial_relative_pose/readout',
                           rr.TextDocument(relative_document(event,roles_for(self.cfg)),media_type='text/markdown'))

    def pose(self,role,stamp,pose,epoch):
        self.time(stamp)
        rec=self.recording
        root=f'/world/{role}/camera'
        path=self.paths[role]
        if pose is None:
            if self.last_valid[role]:
                rec.log(root,rr.Clear(recursive=True))
                path['segment']+=1; path['chunk']=0; path['points']=[]; path['last']=None
            self.last_valid[role]=False
            return
        if path['epoch']!=epoch:
            rec.log(f'/world/trajectories/{role}',rr.Clear(recursive=True))
            path.update(epoch=epoch,segment=0,chunk=0,points=[],entities=deque(),last=None)
        self.last_valid[role]=True
        rec.log(root,rr.Transform3D(translation=pose[:3,3],mat3x3=pose[:3,:3]))
        rec.log(root+'/axes',rr.Arrows3D(origins=np.zeros((3,3)),vectors=np.eye(3)*self.cfg['viewer']['axis_length_m'],colors=AXES))
        rec.log(root+'/origin',rr.Points3D([[0,0,0]],labels=[f'{NAMES[role]} {camera_label(self.cfg,role)}'],colors=COLORS[role],radii=.008))
        p=pose[:3,3].tolist()
        if path['last'] is not None and np.linalg.norm(np.array(p)-path['last'])<self.cfg['viewer']['trajectory_min_distance_m']:
            return
        path['last']=np.array(p)
        path['points'].append(p)
        entity=f'/world/trajectories/{role}/s{path["segment"]}/c{path["chunk"]}'
        if not path['entities'] or path['entities'][-1]!=entity:
            path['entities'].append(entity)
        if len(path['points'])>1:
            rec.log(entity,rr.LineStrips3D([path['points']],colors=COLORS[role],radii=.002))
        length=self.cfg['viewer']['trajectory_chunk_points']
        if len(path['points'])>=length:
            path['chunk']+=1; path['points']=[p]
        while len(path['entities'])*length>self.cfg['viewer']['max_trajectory_points']:
            rec.log(path['entities'].popleft(),rr.Clear(recursive=True))

    def event(self,event):
        self.time(event['time'])
        self.recording.log('/events',rr.TextLog(str(event)))


def relative_document(event,roles):
    """Frozen readout rows for enabled non-reference roles; units in the title."""
    lines=[]
    for role in RELATIVE_ORDER:
        if role not in roles: continue
        if event is None:
            p=r='--, --, --'
        else:
            delta=event['initial_deltas'][role]
            p=', '.join(f'{v:+.3f}' for v in delta['delta_p_m'])
            r=', '.join(f'{v:+.2f}' for v in delta['delta_rpy_deg'])
        lines.append(f'{NAMES[role]:10}  Δp ({p})   ΔR ({r})')
    return '```text\n'+'\n'.join(lines)+'\n```'
