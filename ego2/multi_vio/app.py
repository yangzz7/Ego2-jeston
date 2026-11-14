"""Live coordinator, with bounded asynchronous board processing per device."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import time
import shutil
from types import SimpleNamespace
import cv2
import numpy as np
import zmq
from ..aprilgrid.temporal import Pipeline
from .alignment import Coordinator, inverse
from .recording import Recording
from .transport import Backend, ensure_native


class BoardWorker:
    def __init__(self,data):
        self.pipeline=Pipeline(data['aprilgrid'],data['cameras'],data['board'])
        self.future=None

    def process(self,packet,images,raw_transport=None):
        start=time.perf_counter()
        observations=self.pipeline.process(images,packet['timestamp'])
        return packet,images,observations,copy.deepcopy(self.pipeline.track),(time.perf_counter()-start)*1000,raw_transport

    def close(self):
        try:
            if self.future: self.future.result()
        finally:
            self.pipeline.close()


class Output:
    def __init__(self,cfg,devices,args,recording,viewer):
        self.cfg,self.devices,self.args=cfg,devices,args
        self.roles=tuple(devices)
        self.coordinator=Coordinator(cfg,devices)
        self.recording,self.viewer=recording,viewer
        self.counts={r:dict(frames=0,valid_vio=0,board_frames=0,valid_board=0,world_poses=0) for r in self.roles}
        self.pose_keys={r:None for r in self.roles}
        self.statuses={r:None for r in self.roles}
        self.board_timings={r:deque(maxlen=60) for r in self.roles}
        self.board_support={}
        self.event_index=0

    def ingest(self,role,packet,now):
        self.coordinator.ingest(role,packet,now)
        self.counts[role]['frames']+=1
        self.counts[role]['valid_vio']+=bool(packet['valid'])
        for key in ('firmware','imu_samples','dropped_rgb'):
            if key in packet: self.counts[role][key]=packet[key]

    def board(self,role,result,now):
        packet,images,observations,track,ms=result[:5]
        # A worker finishing after a reset must not re-anchor the new map.
        if self.coordinator.devices[role].map_id!=packet['map_id']: return
        self.counts[role]['board_frames']+=1
        self.counts[role]['valid_board']+=track.measurement is not None
        if hasattr(self,'board_timings'):
            self.board_timings[role].append((now,ms,1000*(now-packet.get('host_monotonic',now))))
        if hasattr(self,'board_support'):
            inliers=track.measurement.inliers if track.measurement is not None else ()
            # Count real image observations only, excluding inferred overlays.
            self.board_support[role]={name:(len({o.marker_id for o in observations
                if o.camera==name and o.reason=='ok' and o.source in ('decoded','flow')}),
                sum(key[0]==name for key in inliers)) for name in ('cam0','cam1')}
        self.coordinator.observe_board(role,packet,track.measurement,now)
        self.recording.board(role,packet,images,observations,track,self.devices[role],self.coordinator.status(role,now),ms)
        if self.viewer:
            if getattr(self.args,'demo',False):
                self.viewer.images(role,packet,images,observations,track,self.coordinator.status(role,now),ms,
                                   raw_transport=result[5] if len(result)>5 else None)
            else:
                self.viewer.board_overlay(role,packet,observations,track,self.coordinator.status(role,now),ms)

    def update(self,now,event_time):
        c=self.coordinator
        c.relative(now)
        poses={r:c.pose(r,now) for r in self.roles}
        for role in self.roles:
            s=c.devices[role]
            status=c.status(role,now)
            if status!=self.statuses[role]:
                # Sample counts can change every frame: terminal progress is
                # reported separately; the viewer still receives each update.
                if self.viewer: self.viewer.state(role,event_time,status)
                self.statuses[role]=status
            pose=poses[role]
            key=(s.packet['frame_id'] if s.packet else -1,s.epoch,pose is not None)
            if key!=self.pose_keys[role]:
                stamp=s.packet['capture_time'] if pose is not None else event_time
                if self.viewer: self.viewer.pose(role,stamp,pose,s.epoch)
                if s.packet and (self.pose_keys[role] is None or key[0]!=self.pose_keys[role][0] or key[2]!=self.pose_keys[role][2]):
                    self.recording.pose(role,{**s.packet,'epoch':s.epoch,'status':status},pose,c.board_pose(role,now))
                self.counts[role]['world_poses']+=pose is not None
                self.pose_keys[role]=key
        if self.viewer:
            self.viewer.initialize_poses(c.initial_poses)
        while self.event_index<len(c.events):
            event=c.events[self.event_index]; self.event_index+=1
            self.recording.event(event)
            if self.viewer: self.viewer.event(event)
            if event['event']=='aligned':
                q=event['quality']
                print(f"[aligned] {event['role']} | {q['inliers']}/{q['total']} samples | position RMS {q['translation_rms_m']*1000:.2f} mm | rotation RMS {q['rotation_rms_deg']:.3f} deg",flush=True)
            elif event['event']=='started':
                print(f'[running] All {len(self.roles)} VIO worlds aligned. Shared world: {c.reference} VIO gravity +Z up; origin at initial board centre. Alignment locked.',flush=True)
                if self.viewer: self.viewer.initial_relative(event)
                for role,delta in c.initial_deltas.items():
                    print(f'[initial {role} relative to {c.reference}] delta p (G, m, XYZ)={delta["delta_p_m"]}; delta R (G, deg, RPY)={delta["delta_rpy_deg"]}',flush=True)
            else: print('[state] '+json.dumps(event),flush=True)

    def progress(self,now):
        print(' | '.join(f'{r} {self.devices[r]["native"]["serial"]}: {self.coordinator.status(r,now)}; frames={self.counts[r]["frames"]}' for r in self.roles),flush=True)
        for role,samples in self.board_timings.items():
            state=self.coordinator.devices[role]
            if state.anchor is not None or not samples: continue
            elapsed=samples[-1][0]-samples[0][0]
            rate=(len(samples)-1)/elapsed if elapsed>0 else 0.
            _,work_ms,age_ms=samples[-1]
            limit=1000*self.cfg['timing']['board_max_age_s']
            print(f'[alignment] {role}: board={work_ms:.0f} ms; total_age={age_ms:.0f}/{limit:.0f} ms; '
                  f'board_rate={rate:.2f} FPS; samples={len(state.window.samples)}/{self.cfg["alignment"]["min_samples"]}',flush=True)
            support=self.board_support.get(role,{})
            tags=' '.join(f'{name}={good}/{observed}' for name,(observed,good) in support.items())
            window=state.window
            span=window.samples[-1][0]-window.samples[0][0] if window.samples else 0.
            print(f'[alignment-quality] {role}: inliers/observed {tags}; need={self.cfg["alignment"]["min_tags_per_eye"]}/eye; '
                  f'span={span:.1f}/{self.cfg["alignment"]["max_duration_s"]:.0f}s; resets={window.resets}; '
                  f'last_reset={window.last_reset}({window.last_reset_samples}); expired={window.expired_samples}',flush=True)


def manual_controller(cfg,devices,args,output,resources,origin,group=None,backends=None,clock=None):
    from .manual import ManualController
    from .storage import status
    recording=output.recording
    def prepare(root,token):
        nonlocal recording
        recording=Recording(cfg,devices,args,prepared_root=root)
        output.recording=recording
        recording.active=False
        for role in devices:
            recording.native_directory(role)
            if backends:
                shutil.copy2(backends[role].directory/'orb.yaml',root/'calibration'/role/'orb.yaml')
        if group: group.prepare(root,token)
    def ready():
        if backends and any(b.process.poll() is not None for b in backends.values()):
            raise RuntimeError('相机后端已退出，无法开始录制。')
        return group.ready() if group else True
    def commit(start):
        recording.cutoff=start
        recording.active=True
        recording.origin=clock() if args.demo else start
        recording.timeline_offset=recording.origin if args.demo else start-origin
        recording.metadata=status(recording.session,'recording',manual_recording=True,
            clock_origin_monotonic=start,record_start_monotonic=start,preview_origin_monotonic=origin,
            recording_timeline_offset=recording.timeline_offset)
        now=clock() if args.demo else time.monotonic()
        c=output.coordinator
        for role,s in c.devices.items():
            if s.packet:
                recording.pose(role,{**s.packet,'epoch':s.epoch,'status':c.status(role,now)},
                               c.pose(role,now),c.board_pose(role,now),seed=True)
        output.pose_keys={r:None for r in devices}
        if group: group.commit(start)
    def abort():
        if group: group.abort()
        for name in ('poses','boards','events'):
            handle=getattr(recording,name,None)
            if handle: handle.close()
        recording.session=None;recording.poses=recording.events=None;recording.active=False
    def stop(end):
        recording.end_cutoff=end
        if group: group.stop(end)
    def stopped():
        if time.monotonic()<recording.end_cutoff: return False
        if backends and any(b.process.poll() is not None for b in backends.values()):
            raise RuntimeError('相机后端在停止录制期间退出，缓存已保留。')
        return group.stopped() if group else True
    def finalize():
        recording.active=False
        c=output.coordinator
        snapshot=SimpleNamespace(**copy.deepcopy({key:getattr(c,key) for key in (
            'shared_from_board','start_time','initial_relatives','initial_deltas','initial_poses')}),
            devices={r:SimpleNamespace(**copy.deepcopy({key:getattr(s,key) for key in (
                'map_id','anchor','aligned_at','epoch')}),
                window=SimpleNamespace(quality=copy.deepcopy(s.window.quality))) for r,s in c.devices.items()})
        counts=copy.deepcopy(output.counts)
        saved_recording=recording
        output.recording=Recording(cfg,devices,SimpleNamespace(**{**vars(args),'record':False}))
        if backends:
            for role,backend in backends.items():
                shutil.copy2(backend.directory/'native.log',saved_recording.work/role/'native.log')
        return lambda:saved_recording.finish(snapshot,counts)
    controller=ManualController('vio',args,prepare,ready,commit,abort,
                                stop=stop,stopped=stopped,finalize=finalize)
    resources.callback(controller.close)
    return controller


def live(cfg,devices,args,output,executable,resources):
    context=zmq.Context()
    resources.callback(context.term)
    directory=resources.enter_context(tempfile.TemporaryDirectory(prefix='ego2-'+cfg['mode']+'-'))
    origin=time.monotonic()
    output.recording.origin=origin
    if output.recording.session:
        from .storage import status
        output.recording.metadata=status(output.recording.session,'recording',clock_origin_monotonic=origin)
    backends={}
    group=None
    if getattr(args,'manual_record',False):
        from .manual import WorkerGroup
        group=WorkerGroup(Path(directory)/'_control',devices)
    for role in devices:
        backend=Backend(role,devices[role]['native'],directory,context,origin,executable,
                        archive=output.recording.native_directory(role),
                        archive_control=group.directories[role] if group else None,
                        display_raw=bool(output.viewer and output.viewer.display_processors.get(role)))
        backends[role]=backend
        def close_backend(backend=backend,role=role):
            backend.close()
            if output.recording.session:
                shutil.copy2(backend.directory/'native.log',output.recording.work/role/'native.log')
                if backend.process.returncode != 0:
                    raise RuntimeError(f'{role}: native archive stopped with code {backend.process.returncode}; cache retained')
        resources.callback(close_backend)
        if output.recording.session:
            shutil.copy2(backend.directory/'orb.yaml',output.recording.session/'calibration'/role/'orb.yaml')
    # Signal every process before waiting for any shutdown.
    resources.callback(lambda:[b.request_stop() for b in backends.values()])
    controller=manual_controller(cfg,devices,args,output,resources,origin,group,backends) if group else None
    workers={}
    for role in devices:
        workers[role]=BoardWorker(devices[role])
        resources.callback(workers[role].close)
    pool=resources.enter_context(ThreadPoolExecutor(max_workers=len(devices),thread_name_prefix='vio-board'))
    poller=zmq.Poller()
    for backend in backends.values(): poller.register(backend.socket,zmq.POLLIN)
    report=origin-2
    while True:
        if controller: controller.poll()
        now=time.monotonic()
        if args.seconds and now-origin>=args.seconds: break
        if args.frames and min(output.counts[r]['frames'] for r in devices)>=args.frames: break
        completed={}
        for role,backend in backends.items():
            worker=workers[role]
            if worker.future and worker.future.done():
                completed[role]=worker.future.result()
                worker.future=None
            if backend.process.poll() is not None and not backend.failed:
                backend.failed=True
                output.coordinator.devices[role].disconnected=True
                print(f'[error] {role} native backend exited:\n{backend.error_tail()}',flush=True)
                if not output.coordinator.started:
                    raise RuntimeError(f'{role} backend exited before initial alignment')
            if backend.failed: continue
            received=backend.receive(need_images=bool(output.viewer) or worker.future is None)
            if received:
                packet,images=received
                output.ingest(role,packet,now)
                if images is not None and output.viewer:
                    output.viewer.preview(role,packet,images,output.coordinator.status(role,now),
                                          raw_transport=backend.read_display_raw(packet))
                if images is not None and worker.future is None:
                    worker.future=pool.submit(worker.process,packet,images)
            elif not output.counts[role]['frames'] and now-origin>cfg['timing']['startup_timeout_s']:
                raise RuntimeError(f'{role}: no frames within startup timeout\n{backend.error_tail()}')
        # Read every available live VIO packet before deciding whether a
        # delayed board result can align it. Detection never blocks capture.
        now=time.monotonic()
        for role,result in completed.items(): output.board(role,result,now)
        output.update(now,now-origin)
        if all(b.failed for b in backends.values()):
            raise RuntimeError('All native backends have stopped')
        if now-report>=2:
            output.progress(now); output.recording.check_size(); report=now
        poller.poll(10)


def demo(cfg,devices,args,output,resources):
    from ..aprilgrid.demo import DemoSource
    from ..common.paths import ROOT
    print('[demo] Synthetic VIO poses + real AprilGrid image detection. NOT a hardware/VIO accuracy test.',flush=True)
    sources,workers,anchors={},{},{}
    texture=cv2.imread(str(ROOT/'tests/fixtures/aprilgrid/calib_io_500x500_6x6_55.png'),cv2.IMREAD_GRAYSCALE)
    for index,role in enumerate(devices):
        d=devices[role]
        source=DemoSource(d['aprilgrid'],d['cameras'],d['board'])
        if texture is not None: source.texture=texture
        source.open(); resources.callback(source.close)
        sources[role]=source
        workers[role]=BoardWorker(d); resources.callback(workers[role].close)
        a=np.eye(4); a[:3,:3]=cv2.Rodrigues(np.array([.1,-.2,.4+index*.5]))[0]; a[:3,3]=[index*.7,.3,-.2]
        anchors[role]=a
    started=time.monotonic()
    shared_from_board=np.eye(4)
    shared_from_board[:3,:3]=anchors[cfg['reference_role']][:3,:3].T
    errors=[]
    stamp=0.
    controller=manual_controller(cfg,devices,args,output,resources,started,clock=lambda:stamp) if getattr(args,'manual_record',False) else None
    for index in range(args.frames or 120):
        stamp=index/10
        if controller: controller.poll()
        if args.seconds and time.monotonic()-started>=args.seconds: break
        entries={}
        for j,role in enumerate(devices):
            source=sources[role]
            camera_board=source.pose_at(stamp*.15)
            camera_board[:3,3]+=[(j-.5)*.12,-j*.08,j*.08]
            truth=inverse(camera_board)
            local=inverse(anchors[role]) @ truth
            packet=dict(protocol=1,serial=devices[role]['native']['serial'],timestamp=stamp,slam_timestamp=stamp,
                capture_time=stamp,host_monotonic=stamp,frame_id=index,map_id=f'demo_{role}',revision=0,
                state=2,imu_initialized=index>=3,initialization_complete=index>=10,valid=index>=10,
                T_world_camera=local.tolist() if index>=10 else None,synthetic_only=True)
            images=tuple(source.render(devices[role]['cameras'][name],camera_board) for name in ('cam0','cam1')) if index<70 else tuple(np.full((1024,1280,3),65,np.uint8) for _ in range(2))
            entries[role]=(packet,images,truth)
            output.ingest(role,packet,stamp)
            output.recording.demo_capture(role,packet,images)
        for role,(packet,images,truth) in entries.items():
            output.board(role,workers[role].process(packet,images),stamp)
        output.update(stamp,stamp)
        for role,(_,_,truth) in entries.items():
            estimated=output.coordinator.pose(role,stamp)
            if estimated is not None:
                expected=shared_from_board @ truth
                errors.append(float(np.linalg.norm(estimated[:3,3]-expected[:3,3])))
        if index%20==0: output.progress(stamp)
        output.recording.check_size()
    if not output.coordinator.started:
        raise RuntimeError('Synthetic regression failed to initialize all board alignments')
    print(f'[demo] world-position max error: {max(errors,default=0)*1000:.2f} mm (synthetic only)',flush=True)


def run(cfg,devices,args):
    if not args.demo:
        from .transport import check_devices
        check_devices(devices)
    executable=None if args.demo else ensure_native(devices[cfg['reference_role']]['native'],args.jobs,args.rebuild)
    cv2.setNumThreads(1)
    recording=Recording(cfg,devices,SimpleNamespace(**{**vars(args),'record':False}) if getattr(args,'manual_record',False) else args)
    error=None
    output=Output(cfg,devices,args,recording,None)
    try:
        with ExitStack() as resources:
            if not args.no_viz and not args.headless:
                from .visualizer import Visualizer
                output.viewer=Visualizer(cfg,devices,args,None)
                resources.callback(output.viewer.close)
            if args.demo: demo(cfg,devices,args,output,resources)
            else: live(cfg,devices,args,output,executable,resources)
    except KeyboardInterrupt:
        print(f'[stop] Closing {len(devices)} native VIO processes...',flush=True)
    except BaseException as exc:
        error=str(exc)
        raise
    finally:
        output.recording.finish(output.coordinator,output.counts,error)
        print('[summary] '+json.dumps(output.counts),flush=True)
        if not output.coordinator.started:
            print('[summary] Initial alignment not completed; no shared-world trajectories were published.',flush=True)
    return output
