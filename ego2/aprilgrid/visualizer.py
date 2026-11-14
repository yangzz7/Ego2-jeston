"""Synchronized native fisheye images, board-relative camera poses and timeline."""
import cv2
import json
import numpy as np
import rerun as rr
import rerun.blueprint as b

from ..common.geometry import project
from ..common.viewer import ViewerConnection, TIMELINE
from .tracking import display_source
from .camera_pose import configure_pose_viewer, camera_pose_frame, pose_document

AXIS_COLORS = [[255, 65, 65], [65, 230, 100], [65, 140, 255]]
REASONS = {"reprojection_outlier": "ERR", "insufficient_inliers": "FEW",
           "stereo_inconsistent": "STEREO", "ambiguous_pose": "AMB",
           "pose_jump": "JUMP", "too_small": "SMALL", "duplicate_id": "DUP",
           "distance_out_of_range": "RANGE", "initialization_failed": "INIT",
           "invalid_projection": "PROJ", "unstable_inliers": "UNSTABLE"}


def build_blueprint(cfg, display_isp=False):
    viewer = configure_pose_viewer(cfg["viewer"])
    rows = []
    for group, title in (("display", "ISP RGB" if display_isp else "RGB"),
                         ("tracking", "ISP + IDs + XYZ" if display_isp else "IDs + XYZ")):
        rows.append(b.Horizontal(*[
            b.Spatial2DView(name=f"{ch['label']} | {title}", origin=f"/{group}/{ch['camera']}",
                            contents=f"/{group}/{ch['camera']}/**") for ch in cfg["channels"]],
            column_shares=[.5, .5]))
    rows.append(b.Horizontal(*[
        b.TextDocumentView(name=f"{ch['label']} | Camera in board",
                           origin=f"/camera_pose/{ch['camera']}",
                           contents=f"/camera_pose/{ch['camera']}/readout")
        for ch in cfg["channels"]], column_shares=[.5, .5]))
    share = viewer["pose_row_share"]
    return b.Blueprint(b.Vertical(*rows, row_shares=[(1-share)/2, (1-share)/2, share]),
        b.TimePanel(state="expanded", timeline=TIMELINE, play_state="following"),
        b.BlueprintPanel(state="hidden"), b.SelectionPanel(state="hidden"),
        auto_layout=False, auto_views=False)


def projected_axes(camera, pose, length):
    """Sample 3D segments: a straight 3D axis curves in a fisheye image."""
    strips = []
    for axis in range(3):
        xyz = pose[:3, 3] + np.linspace(0, length, 20)[:, None]*pose[:3, axis]
        u, v, valid = project(camera, xyz)
        valid &= (u >= 0) & (v >= 0) & (u < camera.size[0]) & (v < camera.size[1])
        # Never connect through an invalid angular region or across the image edge.
        for indices in np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1)+1):
            if len(indices) > 1:
                strips.append((np.c_[u[indices], v[indices]], AXIS_COLORS[axis]))
    return strips


def projected_axes_batch(camera, poses, length):
    """Same ordered strips as projected_axes, projecting all tags at once."""
    if not poses:
        return []
    poses = np.asarray(poses)
    xyz = poses[:, None, None, :3, 3] + np.linspace(0, length, 20)[None, None, :, None]*poses[:, :3, :3].transpose(0, 2, 1)[:, :, None, :]
    u, v, valid = project(camera, xyz)
    valid &= (u >= 0) & (v >= 0) & (u < camera.size[0]) & (v < camera.size[1])
    strips = []
    for tag_u, tag_v, tag_valid in zip(u, v, valid):
        for axis, (x, y, good) in enumerate(zip(tag_u, tag_v, tag_valid)):
            if good.all():
                strips.append((np.column_stack((x, y)), AXIS_COLORS[axis]))
                continue
            indices = np.flatnonzero(good)
            for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1):
                if len(segment) > 1:
                    strips.append((np.column_stack((x[segment], y[segment])), AXIS_COLORS[axis]))
    return strips


class Visualizer(ViewerConnection):
    def __init__(self, cfg, cameras, board, args, save, display_isp=False):
        self.cfg, self.cameras, self.board = cfg, cameras, board
        self.recording = rr.RecordingStream("ego2-aprilgrid-tracking")
        self.blueprint = build_blueprint(cfg, display_isp)
        self.web_process = None
        ref_inverse = np.linalg.inv(cameras[cfg["reference"]].T_from_root)
        self.transforms = {name: cam.T_from_root @ ref_inverse for name, cam in cameras.items()}
        try:
            self._configure(cfg, args.headless, save, args.connect, args.web, args.web_port, args.grpc_port)
        except BaseException:
            self.close()
            raise

    def _image(self, image):
        return rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).compress(
            jpeg_quality=self.cfg["viewer"]["jpeg_quality"])

    def log_raw(self, pair, stamp):
        """Hidden algorithm images, retained at every inference frame for replay."""
        self.recording.set_time(TIMELINE, duration=stamp)
        self.recording.set_time("frame", sequence=pair.sequence)
        for ch, image in zip(self.cfg["channels"], pair.images):
            self.recording.log(f"/raw/{ch['camera']}/image", self._image(image))

    def log(self, pair, stamp, observations, track, processing_ms, diagnostics=None, camera_pose=None,
            *, display_images=None, display_info=None, record_raw=True):
        rec = self.recording
        rec.set_time(TIMELINE, duration=stamp)
        rec.set_time("frame", sequence=pair.sequence)
        if display_info is not None:
            rec.log("/display/isp", rr.TextDocument(json.dumps(display_info, allow_nan=False)))
        measurement = track.measurement
        if camera_pose is None:
            camera_pose = camera_pose_frame(stamp, track, self.transforms, self.cfg["viewer"].get("pose_hold_s", .3))
        images = pair.images if display_images is None else display_images
        for ch, original, image in zip(self.cfg["channels"], pair.images, images):
            name = ch["camera"]
            rec.log(f"/camera_pose/{name}/readout",
                    rr.TextDocument(pose_document(camera_pose, name, self.cfg["viewer"]),
                                    media_type="text/markdown"))
            root = f"/tracking/{name}"
            raw_rgb = self._image(original) if record_raw else None
            if record_raw:
                rec.log(f"/raw/{name}/image", raw_rgb)
            rgb = raw_rgb if record_raw and image is original else self._image(image)
            rec.log(f"/display/{name}/image", rgb)
            rec.log(root+"/image", rgb)
            if display_info is not None:
                if display_info["status"] == "fallback":
                    rec.log(f"/display/{name}/status", rr.Points2D([[image.shape[1]/2, 30]],
                        labels=["ISP unavailable | original RGB"], show_labels=True,
                        colors=[255, 190, 60], radii=rr.Radius.ui_points(1)))
                else:
                    rec.log(f"/display/{name}/status", rr.Clear(recursive=True))
            visible = [o for o in observations if o.camera == name]
            # Replace fixed batches directly. Clearing every active frame can
            # expose an empty intermediate state in a live streaming viewer.
            if not visible:
                rec.log(root+"/overlays", rr.Clear(recursive=True))
            outlines, colors, anchors, labels, poses = [], [], [], [], []
            for obs in visible:
                source = display_source(obs, track)
                accepted = source is not None
                color = [100, 200, 255] if source == "predicted" else [80, 220, 140] if accepted else [255, 190, 60]
                outlines.append(obs.display_outline if obs.display_outline is not None else obs.outline)
                colors.append(color)
                reason = (obs.reason if obs.reason != "ok" else
                          track.status if track.status == "pose_jump" else obs.pose_reason)
                label = (f"{obs.marker_id}~" if source == "predicted" else str(obs.marker_id)) if accepted else f"{obs.marker_id}? {REASONS.get(reason, reason)}"
                pixels = obs.display_pixels if obs.display_pixels is not None else obs.pixels
                anchor = pixels.mean(axis=0)
                anchor[1] = max(12, pixels[:, 1].min()-8)
                anchors.append(anchor)
                labels.append(label)
                if accepted:
                    pose = self.transforms[name] @ self.board.tag_pose(track.display_pose, obs.marker_id)
                    poses.append(pose)
            axes_all = projected_axes_batch(self.cameras[name], poses, self.cfg["viewer"]["axis_length_m"])
            # Batch by eye, avoiding hundreds of small Arrow/log calls per frame.
            if visible:
                rec.log(root+"/overlays/outlines", rr.LineStrips2D(outlines, colors=colors, radii=rr.Radius.ui_points(.7)))
                rec.log(root+"/overlays/ids", rr.Points2D(anchors, labels=labels, show_labels=True,
                                                       colors=colors, radii=rr.Radius.ui_points(1)))
            if axes_all:
                rec.log(root+"/overlays/axes", rr.LineStrips2D([a for a, _ in axes_all], colors=[c for _, c in axes_all],
                                                            radii=rr.Radius.ui_points(1.5)))
            elif visible:
                rec.log(root+"/overlays/axes", rr.Clear(recursive=True))
            count = sum(display_source(o, track) is not None for o in visible)
            status = f"predicting {track.prediction_age_s*1000:.0f} ms" if track.display_predicted else track.status
            text = f"{status} | {count}/{len(visible)} tags | {processing_ms:.0f} ms | XYZ = RGB"
            if display_info is not None:
                text += (f" | ISP {display_info['processing_ms']:.0f} ms" if display_info["status"] == "enhanced"
                         else " | ISP unavailable: original RGB")
            if measurement is None and diagnostics and diagnostics.get("required_inliers"):
                text += f" | support {diagnostics['inliers']}/{diagnostics['required_inliers']} required"
            rec.log(root+"/overlays/status", rr.Points2D([[image.shape[1]/2, 30]], labels=[text], show_labels=True,
                                                       colors=[240, 240, 240], radii=rr.Radius.ui_points(1)))
