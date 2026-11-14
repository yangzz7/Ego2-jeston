"""One LeRobot v3 dataset with named stereo channels and explicit clock matching.

Per-device durable capture is finalized independently of the preview. Assembly
uses every reference-camera frame and nearest peer frames, preserving original
source times, match errors, missing/repeated masks and full-rate IMU sidecars.
"""
from pathlib import Path
import json
import math
import shutil
import hashlib

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from ..gesture.dataset import (_Video, _Moments, _hand_values, _features,
                               DATA_PATH, VIDEO_PATH, EPISODES_PATH, jsonable)
from .storage import write_json, status


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def feature(dtype, n=1):
    return dict(dtype=dtype, shape=[n], names=None)


def features_for(roles):
    out = {k: feature('int64') for k in ('frame_index','episode_index','index','task_index')}
    out.update(timestamp=feature('float32'))
    out['observation.capture_time'] = feature('float64')
    for role in roles:
        p = f'observation.{role}.'
        for k in ('source_timestamp','receive_timestamp','time_offset_s'):
            out[p+k] = feature('float64')
        for k in ('source_sequence','epoch'):
            out[p+k] = feature('int64')
        for k in ('image_valid','repeated','inference_valid','local_pose_valid','shared_pose_valid'):
            out[p+k] = feature('bool')
        for k in ('result_json','map_id'):
            out[p+k] = feature('string')
        for k in ('local_pose','shared_pose'):
            out[p+k] = feature('float64',16)
        for key, f in _features().items():
            if key.startswith('observation.hands.'):
                # These camera names, rather than source lane numbers, are stable.
                for cam in ('cam0','cam1'):
                    if key.startswith('observation.hands.eye0.'):
                        out[p+'hands.'+cam+'.'+key.split('eye0.',1)[1]] = f
    return out


class Writer:
    def __init__(self, root, roles, fps, mode, episode_frames=900):
        self.root, self.roles, self.fps = Path(root), tuple(roles), float(fps)
        self.mode, self.episode_frames = mode, episode_frames
        self.features = features_for(roles)
        self.count = self.episode = 0
        self.batch, self.videos, self.moments = [], {}, {}
        self.global_moments = {}
        self.task = 'Ego2 '+mode
        self.ffmpeg = shutil.which('ffmpeg')
        if not self.ffmpeg:
            raise RuntimeError('Dataset recording requires ffmpeg')
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Invalid dataset FPS')
        (self.root/'meta').mkdir(exist_ok=True)
        for name in ('data','videos'):
            (self.root/name).mkdir(exist_ok=False)

    def append(self, stamp, samples):
        if shutil.disk_usage(self.root).free < 2*1024**3:
            raise RuntimeError('Less than 2 GiB free while finalizing dataset; capture cache retained')
        offset = len(self.batch)
        row = dict(timestamp=offset/self.fps,frame_index=offset,episode_index=self.episode,
                   index=self.count,task_index=0, **{'observation.capture_time':stamp})
        for role, sample in samples.items():
            p = f'observation.{role}.'
            for name in ('source_timestamp','receive_timestamp','source_sequence','epoch','image_valid',
                         'repeated','inference_valid','local_pose_valid','shared_pose_valid','map_id',
                         'time_offset_s','local_pose','shared_pose'):
                row[p+name] = sample[name]
            row[p+'result_json'] = json.dumps(jsonable(sample['result']),ensure_ascii=False,allow_nan=False)
            hands = sample['result'].get('hands',{}) if sample['image_valid'] else {}
            hand_rows = _hand_values([hands.get('cam0',[]),hands.get('cam1',[])])
            for key,value in hand_rows.items():
                row[p+key.removeprefix('observation.').replace('eye0','cam0').replace('eye1','cam1')] = value
            for cam, image in sample['images'].items():
                key = f'observation.images.{role}.{cam}'
                if key not in self.videos:
                    chunk, file = divmod(self.episode,1000)
                    path = self.root/VIDEO_PATH.format(video_key=key,chunk_index=chunk,file_index=file)
                    h,w,_ = image.shape
                    self.features[key] = dict(dtype='video',shape=[h,w,3],names=['height','width','channels'],
                        info={'video.height':h,'video.width':w,'video.fps':self.fps,'video.channels':3,
                              'video.codec':'h264','video.pix_fmt':'gbrp','video.is_depth_map':False,'has_audio':False})
                    self.videos[key] = _Video(path,image.shape,self.fps,self.ffmpeg,isolate_signals=True)
                    self.moments[key] = _Moments()
                self.videos[key].write(image)
                pixels = image[::max(1,image.shape[0]//64),::max(1,image.shape[1]//64),::-1]
                self.moments[key].add(pixels.reshape(-1,3)/255.)
        self.batch.append(row)
        self.count += 1
        if len(self.batch) >= self.episode_frames:
            self.segment()

    def segment(self):
        if not self.batch:
            return
        for video in self.videos.values():
            video.close()
        types = {'int64':pa.int64(),'float32':pa.float32(),'float64':pa.float64(),
                 'bool':pa.bool_(),'string':pa.string()}
        schema = pa.schema([pa.field(k,types[f['dtype']] if f['shape'][0]==1 else
                         pa.list_(types[f['dtype']],f['shape'][0]))
                         for k,f in self.features.items() if f['dtype']!='video'])
        chunk,file = divmod(self.episode,1000)
        path = self.root/DATA_PATH.format(chunk_index=chunk,file_index=file)
        path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(self.batch,schema=schema),path)
        stats = {}
        for key,f in self.features.items():
            if f['dtype']=='string':
                continue
            if f['dtype']=='video':
                stats[key] = self.moments[key].stats(image=True,count=len(self.batch))
            else:
                acc = _Moments();acc.add([r[key] for r in self.batch]);stats[key]=acc.stats()
            self.global_moments.setdefault(key,_Moments()).merge(stats[key])
        episode = dict(episode_index=self.episode,tasks=[self.task],length=len(self.batch),
                       dataset_from_index=self.count-len(self.batch),dataset_to_index=self.count)
        for group in ('data','meta/episodes'):
            episode[group+'/chunk_index']=chunk;episode[group+'/file_index']=file
        for key in self.videos:
            episode.update({f'videos/{key}/chunk_index':chunk,f'videos/{key}/file_index':file,
                            f'videos/{key}/from_timestamp':0.,f'videos/{key}/to_timestamp':len(self.batch)/self.fps})
        for key,fields in stats.items():
            for name,value in fields.items():episode[f'stats/{key}/{name}']=value
        path=self.root/EPISODES_PATH.format(chunk_index=chunk,file_index=file)
        path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist([episode]),path)
        self.episode+=1;self.batch=[];self.videos={};self.moments={}

    def close(self):
        self.segment()
        if not self.count:
            raise RuntimeError('No captured frames; not publishing an empty dataset')
        pd.DataFrame({'task_index':[0]},index=[self.task]).to_parquet(self.root/'meta/tasks.parquet')
        write_json(self.root/'meta/stats.json',{k:v.stats(image=k.startswith('observation.images.'))
                                               for k,v in self.global_moments.items()})
        write_json(self.root/'meta/info.json',dict(codebase_version='v3.0',robot_type='ego2_'+self.mode,
            total_frames=self.count,total_episodes=self.episode,total_tasks=1,fps=self.fps,
            chunks_size=1000,data_files_size_in_mb=100,video_files_size_in_mb=200,
            splits={'train':f'0:{self.episode}'},data_path=DATA_PATH,video_path=VIDEO_PATH,features=self.features))

    def abort(self):
        for video in self.videos.values():
            try:video.close()
            except Exception:pass


def sample(stamp, images, source, received, sequence, result=None):
    return dict(time=stamp,images=images,source_timestamp=float(source),receive_timestamp=float(received),
                source_sequence=int(sequence),epoch=0,image_valid=True,repeated=False,inference_valid=False,
                local_pose_valid=False,shared_pose_valid=False,map_id='',time_offset_s=0.,
                local_pose=[0.]*16,shared_pose=[0.]*16,result=result or {})


class Nearest:
    def __init__(self, source):
        self.source=iter(source);self.before=None;self.after=next(self.source,None);self.last=None
        if self.after is None:raise RuntimeError('One selected camera has no recorded frames')
        self.template=self.after

    def at(self, stamp, tolerance):
        while self.after is not None and self.after['time']<=stamp:
            self.before,self.after=self.after,next(self.source,None)
        candidates=[s for s in (self.before,self.after) if s is not None]
        selected=min(candidates,key=lambda s:abs(s['time']-stamp))
        out=dict(selected,time_offset_s=selected['time']-stamp)
        out['image_valid']=abs(out['time_offset_s'])<=tolerance
        out['repeated']=out['image_valid'] and self.last==out['source_sequence']
        if out['image_valid']:
            self.last=out['source_sequence']
        else:
            out.update(images={c:np.zeros_like(im) for c,im in selected['images'].items()},result={},
                       inference_valid=False,local_pose_valid=False,shared_pose_valid=False,
                       local_pose=[0.]*16,shared_pose=[0.]*16)
        return out


def assemble(root, mode, sources, fps, metadata, *, reference=None, progress=None, total_frames=0):
    root=Path(root)
    reference=reference or ('chest' if 'chest' in sources else next(iter(sources)))
    writer=Writer(root,sources,fps,mode)
    tolerance=.55/fps
    counts={r:dict(missing=0,repeated=0) for r in sources}
    try:
        if progress: progress('生成最终视频',0,total_frames)
        peers={r:Nearest(it) for r,it in sources.items() if r!=reference}
        previous=-math.inf
        for primary in sources[reference]:
            stamp=primary['time']
            if not math.isfinite(stamp) or stamp<previous:raise ValueError('Dataset clock went backwards')
            previous=stamp
            samples={r:primary if r==reference else peers[r].at(stamp,tolerance) for r in sources}
            for role,s in samples.items():
                counts[role]['missing']+=not s['image_valid'];counts[role]['repeated']+=s['repeated']
            writer.append(stamp,samples)
            if progress: progress('生成最终视频',writer.count,total_frames)
        writer.close()
        write_json(root/'meta/ego2.json',dict(schema_version=2,mode=mode,**jsonable(metadata),
            reference_stream=reference,matching_tolerance_s=tolerance,pairing_counts=counts,
            clock='reference-camera samples; nearest peer within tolerance; no hardware synchronization',
            timestamp='episode media frame_index/fps; observation.capture_time is real common receipt time',
            missing='image_valid=false: black image, invalid poses/hands; repeated explicitly marked',
            pose_convention='row-major 4x4 T_A_camera maps optical cam1 into A; metres; shared Z up'))
        return dict(frames=writer.count,episodes=writer.episode,pairing_counts=counts)
    except BaseException:
        writer.abort()
        raise
    finally:
        for source in sources.values():
            if hasattr(source,'close'):source.close()


def gesture_sources(work, roles, origin, demo=False):
    from ..gesture.raw_replay import open_gesture_reader
    readers={r:open_gesture_reader(Path(work)/r/'data') for r in roles}
    def source(reader):
        cams=[c['camera'] for c in reader.profile['channels']]
        for pair,stamp,hands,row in reader.frames():
            item=sample(stamp if demo else pair.received-origin,dict(zip(cams,pair.images)),
                        pair.stamp,pair.received,pair.sequence,
                        dict(hands={cam:jsonable(h) for cam,h in zip(cams,hands)},
                             raw=json.loads(row['observation.gesture_json']),
                             inference_timestamp=row.get('observation.inference_timestamp'),
                             processing_ms=row.get('observation.processing_ms')))
            item['inference_valid']=row['observation.inference_valid']
            yield item
    return readers,{r:source(d) for r,d in readers.items()}


def save_imu(path, samples):
    schema=pa.schema([pa.field('timestamp_seconds',pa.float64()),
                      pa.field('acceleration_m_s2',pa.list_(pa.float64(),3)),
                      pa.field('angular_velocity_rad_s',pa.list_(pa.float64(),3))])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    batch=[];last=-math.inf
    with pq.ParquetWriter(path,schema) as writer:
        for s in samples:
            t=s.get('timestamp_seconds')
            if t is None or t<=last:continue
            last=t
            batch.append({k:s[k] for k in schema.names})
            if len(batch)>=1024:
                writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[]
        if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema))


def finish_gesture(root, work, cfg, origin, demo=False, *, progress=None):
    root,work=Path(root),Path(work)
    if progress: progress('检查录制索引',0,0)
    readers,sources=gesture_sources(work,cfg['devices'],origin,demo)
    if progress: progress('整理 RAW、IMU 和标定',0,len(readers))
    for role,reader in readers.items():
        calibration=reader.root/'meta/calibration'
        if calibration.exists():shutil.copytree(calibration,root/'calibration'/role)
        def imu(reader=reader):
            for c in rows(reader.root/'raw/capture.jsonl'):
                yield from c['metadata'].get('imu_samples',[])
        save_imu(root/'sensors'/role/'imu.parquet',imu())
        # Preserve requested Bayer transport and its original frame index even
        # when a peer frame is not selected by the dataset's common timeline.
        raw=reader.root/'raw/transport'
        if raw.exists():
            shutil.copytree(raw,root/'sensors'/role/'raw/transport')
            shutil.copy2(reader.root/'raw/capture.jsonl',root/'sensors'/role/'raw/capture.jsonl')
    if progress: progress('整理 RAW、IMU 和标定',len(readers),len(readers))
    reference='chest' if 'chest' in readers else next(iter(readers))
    result=assemble(root,'gesture',sources,readers[reference].fps,
        dict(config=cfg,profiles={r:d.profile for r,d in readers.items()},
             devices={r:cfg['devices'][r]['sn'] for r in readers}),reference=reference,
        progress=progress,total_frames=readers[reference].total_frames)
    hashes={role:reader.image_hashes() for role,reader in readers.items() if hasattr(reader,'image_hashes')}
    validate_dataset(root,progress=progress,image_hashes=hashes)
    status(root,'complete',format='lerobot-v3',summary=result)
    shutil.rmtree(work)


def trajectory_sources(work, roles, origin, demo=False):
    work=Path(work)
    def selected(path,role):
        for row in rows(path):
            if row['role']==role:yield row
    def source(role):
        directory=work/role/'native'
        states=iter(selected(work/'poses.jsonl',role));state=next(states,None);current=None
        boards=iter(selected(work/'boards.jsonl',role));board=next(boards,None)
        for packet in rows(directory/'frames.jsonl'):
            seq=packet['frame_id']
            while state is not None and state['frame_id']<=seq:
                if not (current and current['frame_id']==state['frame_id'] and
                        current.get('epoch')==state.get('epoch') and current.get('T_shared_local') is not None
                        and state.get('T_shared_local') is None):
                    current=state
                state=next(states,None)
            while board is not None and board['frame_id']<seq:
                board=next(boards,None)
            annotation=board if board is not None and board['frame_id']==seq else None
            images={cam:cv2.imread(str(directory/name)) for cam,name in packet['images'].items()}
            if any(im is None for im in images.values()):raise RuntimeError('Trajectory image archive is incomplete')
            stamp=(packet['timestamp'] if demo else packet['host_monotonic'])-origin
            item=sample(stamp,images,packet['timestamp'],packet['host_monotonic'],seq,
                        dict(packet={k:v for k,v in packet.items() if k not in ('slot','images')},board=annotation,status=current.get('status','INITIALIZING') if current else 'INITIALIZING'))
            item['map_id']=packet['map_id']
            item['epoch']=current.get('epoch',0) if current else 0
            local=packet.get('T_world_camera')
            if packet['valid'] and local is not None:
                item.update(local_pose_valid=True,local_pose=np.asarray(local).reshape(-1).tolist())
                if current and current['map_id']==packet['map_id'] and current.get('T_shared_local') is not None:
                    shared=np.asarray(current['T_shared_local']) @ np.asarray(local)
                    item.update(shared_pose_valid=True,shared_pose=shared.reshape(-1).tolist())
            yield item
    return {r:source(r) for r in roles}


def finish_trajectory(root,work,cfg,metadata,origin,demo=False):
    root,work=Path(root),Path(work)
    for role in cfg['devices']:
        save_imu(root/'sensors'/role/'imu.parquet',rows(work/role/'native/imu.jsonl'))
    shutil.copy2(work/'events.jsonl',root/'meta/events.jsonl')
    summary=assemble(root,'vio',trajectory_sources(work,cfg['devices'],origin,demo),
        10. if demo else cfg['fps'],dict(config=cfg,devices=metadata['device_roles'],
            initial_deltas=metadata['initial_deltas'],initial_poses=metadata['initial_T_shared_camera'],
            start_time=metadata['start_time']),reference=cfg['reference_role'])
    validate_dataset(root)
    status(root,'complete',format='lerobot-v3',dataset_summary=summary)
    shutil.rmtree(work)


def validate_dataset(root, *, progress=None, image_hashes=None):
    from .replay import Reader
    reader=Reader(root)
    streams={role:iter(rows) for role,rows in (image_hashes or {}).items()}
    current={role:next(stream,None) for role,stream in streams.items()}
    count=0
    if progress: progress('校验最终视频',0,reader.info['total_frames'])
    for _,images,row in reader.frames():
        for role,stream in streams.items():
            if not row[f'observation.{role}.image_valid']: continue
            sequence=row[f'observation.{role}.source_sequence']
            while current[role] is not None and current[role][0]<sequence:
                current[role]=next(stream,None)
            if current[role] is None or current[role][0]!=sequence:
                raise ValueError('Final video references a frame without a captured RGB checksum')
            for camera,image in images[role].items():
                actual=hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest()
                if actual!=current[role][1][camera]:
                    raise ValueError(f'Final video pixels differ from captured RGB: {role}/{camera}, pair {sequence}')
        count+=1
        if progress: progress('校验最终视频',count,reader.info['total_frames'])
    if not count:raise RuntimeError('No replayable dataset rows')
    return count
