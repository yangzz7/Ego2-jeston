"""Opt-in trajectory capture, calibration snapshots and V3 finalization."""
import hashlib
import json
from pathlib import Path
import shutil
import yaml
from .storage import new_session, status
import numpy as np
import cv2
import time


def write_json(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class Recording:
    def __init__(self,cfg,devices,args,prepared_root=None):
        self.session=None
        self.poses=self.events=None
        self.cfg=cfg
        self.active=True
        self.cutoff=None
        self.end_cutoff=None
        self.timeline_offset=0.
        if not args.record: return
        self.session=Path(prepared_root) if prepared_root else new_session('vio', args.session)
        self.work=self.session/'_work'
        self.work.mkdir()
        self.demo=args.demo
        self.origin=0.
        self.boards=(self.work/'boards.jsonl').open('w')
        files={}
        for role in devices:
            directory=self.session/'calibration'/role
            directory.mkdir(parents=True)
            for key in ('path','calibration','imu_calibration'):
                source=Path(devices[role]['native'][key])
                target=directory/('camera.yaml' if key=='path' else key+'.yaml')
                shutil.copy2(source,target)
                if key=='path':
                    bundle=yaml.safe_load(target.read_text())
                    bundle.update(calibration='calibration.yaml',imu_calibration='imu_calibration.yaml')
                    target.write_text(yaml.safe_dump(bundle,sort_keys=False))
                files[str(target.relative_to(self.session))]=hashlib.sha256(target.read_bytes()).hexdigest()
        reference=cfg['reference_role']
        shutil.copy2(devices[reference]['aprilgrid']['target'],self.session/'calibration/target.yaml')
        self.metadata=dict(version=4,mode='vio',status='recording',finished=False,synthetic_only=args.demo,config=cfg,sha256=files,
            reference_role=reference,reference_serial=cfg['devices'][reference]['sn'],
            device_roles={r:cfg['devices'][r]['sn'] for r in devices},
            calibration_serials={r:data['native']['serial'] for r,data in devices.items()},
            pose_convention='T_A_B maps B into A; optical reference cam1; metres; quaternion xyzw',
            board_frame='grid centre; X print-right, Y print-up, Z out of printed face',
            shared_frame=f'initial board centre; axes of {reference} VIO at first alignment; +Z gravity-up; fixed for session',
            pose_fields=dict(T_world_camera='native local VIO frame (legacy field name)',
                             T_board_camera='original board frame',T_shared_camera='shared gravity-aligned Z-up frame, used by Rerun'),
            clock='arrival-based lower-envelope host monotonic mapping, NOT hardware exposure synchronization',
            initial_delta_convention='at common start time; delta_p=p_G_target-p_G_reference; delta_R=R_G_target*R_G_reference^T; reference=reference_role; R=Rz(yaw)Ry(pitch)Rx(roll); singular yaw=0; frozen for session',
            recording='LeRobot v3.0; all native TrackStereo outputs archived before lossy preview; IMU accepted samples independently archived')
        write_json(self.session/'session.json',self.metadata)
        self.metadata=status(self.session,'preparing' if prepared_root else 'recording')
        self.poses=(self.work/'poses.jsonl').open('w')
        self.events=(self.work/'events.jsonl').open('w')
        print(f'[{"record-prepare" if prepared_root else "recording"}] {self.session}',flush=True)

    def accepts(self, packet=None):
        if not self.session or not self.active: return False
        if self.cutoff is None and self.end_cutoff is None: return True
        stamp = time.monotonic() if self.demo or packet is None else packet['host_monotonic']
        return self.session and self.active and (self.cutoff is None or stamp >= self.cutoff) and (
            self.end_cutoff is None or stamp < self.end_cutoff)

    def pose(self,role,packet,pose,board_pose,*,seed=False):
        if self.poses and (seed or self.accepts(packet)):
            self.poses.write(json.dumps(dict(role=role,**packet,
                T_board_camera=board_pose.tolist() if board_pose is not None else None,
                T_shared_camera=pose.tolist() if pose is not None else None,
                T_shared_local=(pose @ np.linalg.inv(np.asarray(packet['T_world_camera']))).tolist()
                    if pose is not None and packet.get('T_world_camera') is not None else None),allow_nan=False)+'\n')

    def event(self,event):
        if self.events and self.accepts():
            self.events.write(json.dumps(event,allow_nan=False)+'\n'); self.events.flush()

    def check_size(self):
        if self.session:
            size=sum(p.stat().st_size for p in self.session.rglob('*') if p.is_file())
            if size>self.cfg['recording']['max_gib']*1024**3:
                raise RuntimeError('Recording size limit reached; stopping acquisition cleanly')

    def finish(self,coordinator,summary,error=None):
        if not self.session: return
        for handle in (self.poses,self.events,self.boards): handle.close()
        (self.session/'meta').mkdir(exist_ok=True)
        write_json(self.session/'meta/alignment.json',{r:dict(map_id=s.map_id,
            T_board_local=s.anchor.tolist() if s.anchor is not None else None,quality=s.window.quality,
            T_shared_local=(coordinator.shared_from_board @ s.anchor).tolist()
                if coordinator.shared_from_board is not None and s.anchor is not None else None,
            aligned_at=s.aligned_at,epoch=s.epoch) for r,s in coordinator.devices.items()})
        self.metadata.update(json.loads((self.session/'session.json').read_text()))
        self.metadata.update(finished=False,status='incomplete' if error else 'finalizing',error=error,summary=summary,
            start_time=coordinator.start_time-self.timeline_offset if coordinator.start_time is not None else None,
            initial_T_reference_camera={r:t.tolist() for r,t in coordinator.initial_relatives.items()},
            initial_deltas=coordinator.initial_deltas,
            initial_T_shared_camera={r:t.tolist() for r,t in coordinator.initial_poses.items()},
            T_shared_board=coordinator.shared_from_board.tolist() if coordinator.shared_from_board is not None else None)
        write_json(self.session/'session.json',self.metadata)

        if error:
            status(self.session,'incomplete',error=error)
            return
        try:
            from .dataset import finish_trajectory
            finish_trajectory(self.session,self.work,self.cfg,self.metadata,self.origin,self.demo)
            print(f'[saved] {self.session}',flush=True)
        except BaseException as exc:
            status(self.session,'incomplete',error=str(exc))
            raise

    def native_directory(self, role):
        if not self.session:return None
        path=self.work/role/'native';path.mkdir(parents=True,exist_ok=True)
        return path

    def demo_capture(self, role, packet, images):
        if not self.accepts(packet):return
        directory=self.native_directory(role)
        item=dict(packet,images={})
        for cam,image in zip(('cam0','cam1'),images):
            name=f'{packet["frame_id"]}_{cam}.png'
            if not cv2.imwrite(str(directory/name),image):raise RuntimeError('Cannot write demo archive')
            item['images'][cam]=name
        with (directory/'frames.jsonl').open('a') as stream:
            stream.write(json.dumps(item)+'\n')
        (directory/'imu.jsonl').touch()

    def board(self, role, packet, images, observations, track, data, status_text, processing_ms):
        if not self.accepts(packet):return
        from ..aprilgrid.tracking import display_source
        from ..aprilgrid.visualizer import projected_axes, REASONS
        cameras={}
        root=np.linalg.inv(data['cameras']['cam1'].T_from_root)
        for cam in ('cam0','cam1'):
            tags=[]
            for obs in observations:
                if obs.camera!=cam:continue
                source=display_source(obs,track)
                color=[100,200,255] if source=='predicted' else [80,220,140] if source else [255,190,60]
                reason=obs.reason if obs.reason!='ok' else obs.pose_reason
                label=(f'{obs.marker_id}~' if source=='predicted' else str(obs.marker_id)) if source else f'{obs.marker_id}? {REASONS.get(reason,reason)}'
                pixels=obs.display_pixels if obs.display_pixels is not None else obs.pixels
                outline=obs.display_outline if obs.display_outline is not None else obs.outline
                axes=[]
                if source:
                    pose=data['cameras'][cam].T_from_root @ root @ data['board'].tag_pose(track.display_pose,obs.marker_id)
                    axes=[(a.tolist(),c) for a,c in projected_axes(data['cameras'][cam],pose,data['aprilgrid']['viewer']['axis_length_m'])]
                tags.append(dict(pixels=pixels.tolist(),outline=outline.tolist(),label=label,color=color,axes=axes))
            cameras[cam]=tags
        self.boards.write(json.dumps(dict(role=role,frame_id=packet['frame_id'],cameras=cameras,
            status=status_text,board_status=track.status,processing_ms=processing_ms))+'\n')
