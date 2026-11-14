"""Exactly one point-cloud view, two unannotated RGB views and a timeline."""
import cv2
import rerun as rr
import rerun.blueprint as b
from types import SimpleNamespace

from ..common.viewer import ViewerConnection, TIMELINE


def build_blueprint():
    return b.Blueprint(
        b.Horizontal(
            b.Spatial3DView(name="3D point cloud", origin="/world", contents="/world/maps/**",
                            line_grid=False, background=[20,24,30]),
            b.Vertical(
                b.Spatial2DView(name="C1 / cam0 RGB", origin="/rgb/cam0", contents="/rgb/cam0"),
                b.Spatial2DView(name="C2 / cam1 RGB", origin="/rgb/cam1", contents="/rgb/cam1"),
                row_shares=[1,1]),
            column_shares=[0.7,0.3]),
        b.TimePanel(state="collapsed",timeline=TIMELINE,play_state="following"),
        b.BlueprintPanel(state="hidden"),b.SelectionPanel(state="hidden"),
        auto_layout=False,auto_views=False,
    )


class Visualizer(ViewerConnection):
    def __init__(self, cfg, args, save):
        self.recording = rr.RecordingStream("ego2-s200032-reconstruction")
        self.blueprint = build_blueprint()
        self.web_process = None
        self.display_processor = None
        self.file_viewer = None
        self.point_radius = float(cfg["viewer"].get("point_radius_px",0.65))
        try:
            from ..common.display_isp import create_display_isp, live_display
            if save and live_display(args):
                options = SimpleNamespace(**{**vars(args),'headless':True,'connect':None,'web':False})
                self.file_viewer = Visualizer(cfg,options,save)
                save = None
            self._configure(cfg,args.headless,save,args.connect,args.web,args.web_port,args.grpc_port)
            self.display_processor = create_display_isp(cfg,cfg['serial'],args)
        except BaseException:
            self.close()
            raise

    def time(self, stamp):
        self.recording.set_time(TIMELINE,duration=max(0,float(stamp)))
        if getattr(self,'file_viewer',None): self.file_viewer.time(stamp)

    @property
    def wants_display_raw(self):
        return self.display_processor is not None

    def rgb(self, images, stamp, *, raw_transport=None):
        if getattr(self,'file_viewer',None): self.file_viewer.rgb(images,stamp)
        if getattr(self,'display_processor',None):
            # Native reconstruction's presentation order is cam1,cam0.
            self.display_processor.submit(raw_transport,images,lambda shown:self._rgb(shown,stamp),order=(1,0))
        else:
            self._rgb(images,stamp)

    def _rgb(self, images, stamp):
        self.recording.set_time(TIMELINE,duration=max(0,float(stamp)))
        for cam,image in zip(("cam1","cam0"),images):
            self.recording.log(f"/rgb/{cam}",rr.Image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).compress(jpeg_quality=90))

    def pose(self, map_id, key, pose):
        if getattr(self,'file_viewer',None): self.file_viewer.pose(map_id,key,pose)
        self.recording.log(f"/world/maps/m{map_id}/k{key}",
                           rr.Transform3D(translation=pose[:3,3],mat3x3=pose[:3,:3]))

    def chunk(self, map_id, key, pose, points, colors):
        if getattr(self,'file_viewer',None):
            self.file_viewer.chunk(map_id,key,pose,points,colors)
        self.recording.log(f"/world/maps/m{map_id}/k{key}",
                           rr.Transform3D(translation=pose[:3,3],mat3x3=pose[:3,:3]))
        self.recording.log(f"/world/maps/m{map_id}/k{key}/points",rr.Points3D(
            points,colors=colors,radii=rr.Radius.ui_points(self.point_radius),point_shading="flat"))

    def remove_chunk(self, map_id, key):
        if getattr(self,'file_viewer',None): self.file_viewer.remove_chunk(map_id,key)
        self.recording.log(f"/world/maps/m{map_id}/k{key}",rr.Clear(recursive=True))

    def close(self):
        if getattr(self,'display_processor',None):
            self.display_processor.close()
            self.display_processor = None
        if getattr(self,'file_viewer',None):
            self.file_viewer.close()
            self.file_viewer = None
        super().close()
