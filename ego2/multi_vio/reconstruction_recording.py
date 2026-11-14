"""Record a running dense map from its current visible state, without restart."""
import json
from types import SimpleNamespace
import time

from .manual import ManualController
from .storage import status, finish_replay


class ReconstructionRecording:
    def __init__(self,cfg,profile,args):
        from ..reconstruction.visualizer import Visualizer
        self.cfg,self.profile,self.args=cfg,profile,args
        self.preview=Visualizer(profile,args,None) if not args.no_viz else None
        self.file=None;self.dense=None;self.start_time=None;self.offset=None;self.end_time=None
        self.stamp=0.;self.snapshot_pending=False
        self.controller=ManualController('reconstruction',args,self.prepare,lambda:True,self.commit,self.abort,
            stop=self.stop,stopped=lambda:time.monotonic()>=self.end_time,finalize=self.finalize)

    def prepare(self,root,token):
        from ..reconstruction.visualizer import Visualizer
        status(root,'preparing',mode='reconstruction',config=self.cfg,
               devices={r:d['sn'] for r,d in self.cfg['devices'].items()},synthetic_only=self.args.demo)
        options=SimpleNamespace(**{**vars(self.args),'headless':True,'web':False,'connect':None})
        self.file=Visualizer(self.profile,options,root/'replay.rrd')

    def commit(self,start):
        self.start_time=start;self.snapshot_pending=True;self.offset=None;self.end_time=None

    def abort(self):
        if self.file:self.file.close()
        self.file=None;self.start_time=None;self.offset=None;self.end_time=None

    def stop(self, end):
        self.end_time=end

    def finalize(self):
        file,self.file=self.file,None
        root,offset=self.controller.root,self.offset
        self.start_time=self.offset=self.end_time=None
        self.snapshot_pending=False
        def finish():
            file.close()
            status(root,'finalizing',preview_time_at_record_start=offset)
            finish_replay(root,root/'replay.rrd',None)
        return finish

    def poll(self):
        self.controller.poll()

    def time(self,stamp):
        self.stamp=stamp
        if self.preview:self.preview.time(stamp)
        if self.start_time is None or time.monotonic()<self.start_time or (
                self.end_time is not None and time.monotonic()>=self.end_time):return
        if self.offset is None:self.offset=stamp
        self.file.time(stamp-self.offset)
        if self.snapshot_pending:
            # DenseMap is the owner of the visible set and cached depths. Emit
            # only that set: removed/reset maps must not reappear in replay.
            if self.dense:
                for key,(map_id,pose) in self.dense.live.items():
                    depth,color=self.dense.depth(key)
                    points,colors=self.dense.stereo.points(depth,color)
                    self.file.chunk(map_id,key,pose,points,colors)
            self.snapshot_pending=False

    def enabled(self):
        return self.file is not None and self.offset is not None and (
            self.end_time is None or time.monotonic()<self.end_time)

    @property
    def wants_display_raw(self):
        return bool(self.preview and self.preview.wants_display_raw)

    def rgb(self,images,stamp,*,raw_transport=None):
        self.time(stamp)
        if self.preview:self.preview.rgb(images,stamp,raw_transport=raw_transport)
        if self.enabled():self.file.rgb(images,stamp-self.offset)

    def pose(self,*args):
        if self.preview:self.preview.pose(*args)
        if self.enabled():self.file.pose(*args)

    def chunk(self,*args):
        if self.preview:self.preview.chunk(*args)
        if self.enabled():self.file.chunk(*args)

    def remove_chunk(self,*args):
        if self.preview:self.preview.remove_chunk(*args)
        if self.enabled():self.file.remove_chunk(*args)

    def close(self):
        if self.preview:self.preview.close();self.preview=None
        if self.file:self.file.close();self.file=None

    def finish(self,error=None):
        self.controller.wait_finalization()
        self.close()
        self.controller.close()
        root=self.controller.root
        if self.controller.active:
            if error:status(root,'incomplete',error=str(error))
            else:
                try:
                    status(root,'finalizing',preview_time_at_record_start=self.offset)
                    finish_replay(root,root/'replay.rrd',None)
                    print(f'[saved] {root}',flush=True)
                except Exception as exc:
                    status(root,'incomplete',error=str(exc));raise
            return json.loads((root/'session.json').read_text())
        return dict(finished=False,status='preview_only')
