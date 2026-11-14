"""One 3D trajectory view, two raw RGB images, and the capture timeline."""
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as b
from ..common.viewer import ViewerConnection, TIMELINE


def build_blueprint(cfg):
    return b.Blueprint(
        b.Horizontal(
            b.Spatial3DView(name=f"Wrist endpoint trajectory / head={cfg['reference']}", origin="/head",
                            contents="/head/**", line_grid=False, background=[20, 24, 30],
                            eye_controls=b.EyeControls3D(position=[0, -0.12, -0.10],
                                                        look_target=[0, 0.06, 0.32], eye_up=[0, -1, 0])),
            b.Vertical(*[b.Spatial2DView(name=f"{ch['label']} RGB", origin=f"/rgb/{ch['camera']}",
                                         contents=f"/rgb/{ch['camera']}") for ch in cfg["channels"]]),
            column_shares=[0.7, 0.3]),
        b.TimePanel(state="collapsed", timeline=TIMELINE, play_state="following"),
        b.BlueprintPanel(state="hidden"), b.SelectionPanel(state="hidden"),
        auto_layout=False, auto_views=False)


class Visualizer(ViewerConnection):
    def __init__(self, cfg, args, save):
        self.cfg = cfg
        self.recording = rr.RecordingStream("ego2-wrist-endpoint-trajectory")
        self.blueprint = build_blueprint(cfg)
        self.web_process = None
        self.axis = cfg["viewer"]["axis_length_m"]
        try:
            self._configure(cfg, args.headless, save, args.connect, args.web, args.web_port, args.grpc_port)
            self.recording.log("/head", rr.ViewCoordinates.RDF, static=True)
        except BaseException:
            self.close()
            raise

    def axes(self, path, poses, length, radius):
        poses = list(poses)
        if not poses:
            return
        origins = np.concatenate([np.tile(p[:3, 3], (3, 1)) for p in poses])
        vectors = np.concatenate([p[:3, :3].T*length for p in poses])
        colors = np.tile([[255, 80, 80], [80, 220, 100], [80, 140, 255]], (len(poses), 1))
        self.recording.log(path, rr.Arrows3D(origins=origins, vectors=vectors, colors=colors, radii=radius))

    def log(self, pair, stamp, tracks):
        self.recording.set_time(TIMELINE, duration=stamp)
        self.recording.set_time("frame", sequence=pair.sequence)
        for ch, image in zip(self.cfg["channels"], pair.images):
            self.recording.log(f"/rgb/{ch['camera']}", rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=90))
        for name, track in tracks.items():
            root = f"/head/hands/{name}"
            strips = track.strips()
            if strips:
                self.recording.log(root+"/trajectory", rr.LineStrips3D(strips, colors=track.target["color"], radii=0.001))
            self.axes(root+"/past_orientations", track.history, self.axis*0.45, 0.0005)
            if track.pose is None:
                self.recording.log(root+"/current", rr.Clear(recursive=True))
                continue
            pose = track.pose
            self.axes(root+"/current/axes", [pose], self.axis, 0.001)
            half = track.target["size_m"] / 2
            local = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
            vertices = local @ pose[:3, :3].T + pose[:3, 3]
            self.recording.log(root+"/current/plane", rr.Mesh3D(
                vertex_positions=vertices, triangle_indices=[[0, 1, 2], [0, 2, 3]],
                vertex_colors=[*track.target["color"], 100]))
            self.recording.log(root+"/current/outline", rr.LineStrips3D(
                [np.vstack([vertices, vertices[0]])], colors=track.target["color"], radii=0.0006))
            self.recording.log(root+"/current/origin", rr.Points3D(
                [pose[:3, 3]], colors=track.target["color"], radii=rr.Radius.ui_points(2),
                labels=[f"{name} / IDs {','.join(map(str, track.measurement.marker_ids))} / {track.status}"], show_labels=True))
