"""Offline Rerun playback/export of unified LeRobot v3 or RRD sessions."""
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pyarrow.parquet as pq
import rerun as rr

from ..common.sources import Pair
from ..common.viewer import ViewerConnection, TIMELINE
from ..gesture.replay import _Video, _hands
from ..gesture.visualizer import Visualizer as GestureVisualizer
from .mode_viewer import build_blueprint as optical_blueprint
from .visualizer import Visualizer as TrajectoryVisualizer, build_blueprint as trajectory_blueprint, draw_compact_id
from .storage import validate_rrd
from .replay_display import ReplayDisplay, ReplayFrames


class Reader:
    def __init__(self, root):
        self.root=Path(root).expanduser().resolve()
        self.info=json.loads((self.root/'meta/info.json').read_text())
        self.meta=json.loads((self.root/'meta/ego2.json').read_text())
        if self.info.get('codebase_version')!='v3.0' or self.meta.get('schema_version')!=2:
            raise ValueError('Expected an Ego2 unified LeRobot v3.0 dataset')
        self.roles=tuple(self.meta['devices'])
        self.fps=float(self.info['fps'])
        self.video_keys=[f'observation.images.{r}.{cam}' for r in self.roles for cam in ('cam0','cam1')]

    def path(self, template, **values):
        path=(self.root/self.info[template].format(**values)).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Dataset path escapes session directory')
        return path

    def frames(self):
        count=0;previous=-np.inf
        for path in sorted((self.root/'meta/episodes').glob('chunk-*/file-*.parquet')):
            for ep in pq.read_table(path).to_pylist():
                number=int(ep['episode_index']);length=int(ep['length']);first=int(ep['dataset_from_index'])
                if first!=count or ep['dataset_to_index']-first!=length:
                    raise ValueError('Broken LeRobot episode index')
                with ExitStack() as resources:
                    videos={}
                    for key in self.video_keys:
                        start=float(ep[f'videos/{key}/from_timestamp'])*self.fps
                        end=float(ep[f'videos/{key}/to_timestamp'])*self.fps
                        if abs(start-round(start))>.01 or abs(end-start-length)>.01:
                            raise ValueError('Video offsets do not match episode length')
                        file=self.path('video_path',video_key=key,chunk_index=ep[f'videos/{key}/chunk_index'],
                                       file_index=ep[f'videos/{key}/file_index'])
                        videos[key]=_Video(file,self.fps,round(start),length,tuple(self.info['features'][key]['shape']))
                        resources.callback(videos[key].close)
                    file=self.path('data_path',chunk_index=ep['data/chunk_index'],file_index=ep['data/file_index'])
                    offset=0
                    for batch in pq.ParquetFile(file).iter_batches(batch_size=64):
                        for row in batch.to_pylist():
                            if row['index']<first or row['index']>=ep['dataset_to_index']:continue
                            if (row['index']!=count or row['frame_index']!=offset or row['episode_index']!=number
                                or abs(row['timestamp']-offset/self.fps)>max(1e-4,.01/self.fps)):
                                raise ValueError('Video/data frame index mismatch')
                            stamp=row['observation.capture_time']
                            if not np.isfinite(stamp) or stamp<previous:raise ValueError('Invalid capture timeline')
                            previous=stamp
                            images={r:{cam:videos[f'observation.images.{r}.{cam}'].read() for cam in ('cam0','cam1')}
                                    for r in self.roles}
                            count+=1;offset+=1
                            yield stamp,images,row
                    if offset!=length:raise ValueError('Truncated episode')
        if count!=self.info['total_frames']:raise ValueError('Dataset total_frames mismatch')


class MappedRecording:
    def __init__(self, recording, role):self.recording,self.role=recording,role
    def set_time(self,name,**value):
        self.recording.set_time(name if name==TIMELINE else self.role+'_'+name,**value)
    def log(self,path,*items,**kwargs):
        for eye,cam in enumerate(('cam0','cam1')):
            path=path.replace(f'/head/eye{eye}',f'/cameras/{self.role}/{cam}')
        self.recording.log(path,*items,**kwargs)


class ReplayViewer(ViewerConnection):
    def __init__(self, reader, destination):
        self.reader=reader;cfg=reader.meta['config']
        self.mode=reader.meta['mode']
        self.recording=rr.RecordingStream('ego2-'+self.mode+'-replay')
        # Raw IMU plots are a live gesture preview. Keep the existing replay
        # layout until its archived device clocks are explicitly mapped here.
        self.blueprint=trajectory_blueprint(cfg) if self.mode=='vio' else optical_blueprint(cfg, imu=False)
        self.web_process=None
        self.display=None
        self._configure(cfg,True,destination,None,False,9090,9876)
        self.recording.log('/session',rr.TextDocument(json.dumps(reader.meta,ensure_ascii=False)),static=True)
        if self.mode=='vio':
            v=TrajectoryVisualizer.__new__(TrajectoryVisualizer)
            v.cfg=cfg;v.roles=reader.roles;v.recording=self.recording;v.blueprint=self.blueprint
            v.eye_initialized=False;v.last_valid={r:False for r in reader.roles}
            v.paths={r:dict(epoch=-1,segment=0,chunk=0,points=[],entities=deque(),last=None) for r in reader.roles}
            self.vio=v
            self.recording.log('/world',rr.ViewCoordinates.RIGHT_HAND_Z_UP,static=True)
            v.time(0.);v.initial_relative(None)
            self.initial_shown=False
        else:
            self.gesture={}
            self.frames={}
            for role in reader.roles:
                v=GestureVisualizer.__new__(GestureVisualizer)
                v.recording=MappedRecording(self.recording,role);v.jpeg_quality=90
                self.gesture[role]=v
                self.frames[role]=ReplayFrames()
        self.display=ReplayDisplay(reader)

    def close(self):
        try:
            if getattr(self,'display',None) is not None:
                self.display.close()
        finally:
            super().close()

    def frame(self, stamp, images, row):
        self.recording.reset_time();self.recording.set_time(TIMELINE,duration=stamp)
        if self.mode=='vio' and not self.initial_shown:
            start=self.reader.meta.get('start_time')
            if start is not None and stamp>=start:
                self.vio.initialize_poses({r:np.asarray(p) for r,p in self.reader.meta['initial_poses'].items()})
                self.vio.initial_relative(dict(time=stamp,initial_deltas=self.reader.meta['initial_deltas']))
                self.initial_shown=True
        for role in self.reader.roles:
            p=f'observation.{role}.';result=json.loads(row[p+'result_json'])
            valid=row[p+'image_valid']
            if self.mode=='gesture':
                draw_frame,held_status=self.frames[role].select(row,p)
                if not draw_frame:
                    # Update only explanatory text. Image and skeleton both
                    # remain at the last matched source frame on the timeline.
                    self.recording.set_time(role+'_frame',sequence=self.frames[role].saved[1])
                    for cam,image in images[role].items():
                        self.recording.log(f'/cameras/{role}/{cam}/status',rr.Points2D(
                            [[image.shape[1]/2,35]],radii=0,labels=[held_status],
                            colors=[240,240,240],show_labels=True,draw_order=30))
                    continue
                hands=result.get('hands',{})
                restored=_hands(json.dumps(dict(hands=[hands.get('cam0',[]),hands.get('cam1',[])])),
                                row[p+'inference_valid']) if valid else [[],[]]
                shown=self.display.images(role,row,images[role])
                pair=Pair(tuple(shown[c] for c in ('cam0','cam1')),row[p+'source_timestamp'],
                          row[p+'receive_timestamp'],row[p+'source_sequence'])
                self.gesture[role].log(pair,stamp,restored,inference_ran=row[p+'inference_valid'])
            else:
                self.vio.pose(role,stamp,np.asarray(row[p+'shared_pose']).reshape(4,4)
                              if row[p+'shared_pose_valid'] else None,row[p+'epoch'])
                board=result.get('board') or {}
                for cam,image in images[role].items():
                    root=f'/cameras/{role}/{cam}';display=image.copy()
                    self.recording.log(root+'/overlays',rr.Clear(recursive=True))
                    tags=board.get('cameras',{}).get(cam,[]) if valid else []
                    for n,tag in enumerate(tags):
                        draw_compact_id(display,np.asarray(tag['pixels']),tag['label'],tag['color'])
                        self.recording.log(root+f'/overlays/{n}/box',rr.LineStrips2D([tag['outline']],colors=tag['color'],radii=.7))
                        if tag['axes']:
                            self.recording.log(root+f'/overlays/{n}/axes',rr.LineStrips2D(
                                [a for a,c in tag['axes']],colors=[c for a,c in tag['axes']],radii=1.5))
                    self.recording.log(root+'/image',rr.Image(cv2.cvtColor(display,cv2.COLOR_BGR2RGB)).compress(jpeg_quality=90))
                    self.recording.log(root+'/status',rr.Points2D([[image.shape[1]/2,35]],radii=0,
                        labels=[result.get('status','NO POSE')],show_labels=True))
            if not valid:
                for cam,image in images[role].items():
                    self.recording.log(f'/cameras/{role}/{cam}/status',rr.Points2D(
                        [[image.shape[1]/2,35]],radii=0,labels=['NO MATCHED FRAME'],show_labels=True))


def export_dataset(root, destination, *, progress=None):
    destination=Path(destination).expanduser().resolve()
    if destination.exists():raise FileExistsError(f'Replay already exists: {destination}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.partial')
    if temporary.exists():raise FileExistsError(f'Partial export already exists: {temporary}')
    reader=Reader(root);viewer=ReplayViewer(reader,temporary)
    count=0
    try:
        if progress: progress(0,reader.info['total_frames'])
        for stamp,images,row in reader.frames():
            viewer.frame(stamp,images,row);count+=1
            if progress and (count % 30 == 0 or count == reader.info['total_frames']):
                progress(count,reader.info['total_frames'])
    finally:
        viewer.close()
    validate_rrd(temporary)
    temporary.replace(destination)
    print(f'[replay] LeRobot v3.0: {count} synchronized rows -> {destination}',flush=True)
    return destination


def replay(args):
    root=Path(args.session).expanduser().resolve()
    metadata=json.loads((root/'session.json').read_text())
    if metadata.get('status')!='complete':
        print('[replay] 此数据未完整保存；只尝试读取已经完成的内容。',flush=True)
    with ExitStack() as resources:
        if (root/'meta/info.json').exists():
            target=Path(args.save).expanduser().resolve() if args.save else Path(
                resources.enter_context(tempfile.TemporaryDirectory(prefix='ego2-replay-')))/'replay.rrd'
            path=export_dataset(root,target)
        else:
            path=root/'replay.rrd'
            validate_rrd(path)
            if args.save:
                target=Path(args.save).expanduser().resolve()
                if target.exists():raise FileExistsError(f'Replay already exists: {target}')
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);path=target
        if not args.headless:
            # Native file playback loads on demand, retaining seekable history
            # even when the recording exceeds the live viewer memory budget.
            subprocess.run(['rerun',str(path)],check=True)
        return path
