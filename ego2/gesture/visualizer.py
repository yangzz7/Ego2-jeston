"""Four gesture views; shared viewer transport lives in ego2.common.viewer."""
from pathlib import Path
import json
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as b
from ..common.viewer import ViewerConnection, TIMELINE

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
               (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def build_blueprint(channels, *, metadata=False):
    rows = []
    for overlay in (False, True):
        views = []
        for i, channel in enumerate(channels):
            origin = f"/head/eye{i}"
            contents = [f"{origin}/rgb"]
            if overlay:
                contents += [f"{origin}/hands/**", f"{origin}/status"]
            views.append(b.Spatial2DView(
                origin=origin, contents=contents,
                name=f"{channel.get('label', channel['camera'])} | {'Tracking' if overlay else 'RGB'}",
            ))
        rows.append(b.Horizontal(*views, column_shares=[1, 1]))
    if metadata:
        rows.append(b.Horizontal(
            b.TimeSeriesView(origin="/sensors/imu", name="IMU (device samples)"),
            b.TextDocumentView(origin="/recording", contents=["/recording/frame"], name="Frame metadata"),
        ))
    shares = [2, 2, 1] if metadata else [1, 1]
    return b.Blueprint(
        b.Vertical(*rows, row_shares=shares),
        b.TimePanel(state="expanded", timeline=TIMELINE, play_state="following"),
        b.BlueprintPanel(state="hidden"), b.SelectionPanel(state="hidden"),
        auto_layout=False, auto_views=False,
    )


class Visualizer(ViewerConnection):
    def __init__(self, profile, *, headless=False, save=None, connect=None,
                 web=False, web_port=9090, grpc_port=9876):
        self.recording = rr.RecordingStream("ego2-s200032-gesture")
        self.jpeg_quality = int(profile.get("viewer", {}).get("jpeg_quality", 95))
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError("viewer.jpeg_quality must be 0 (raw RGB) or 1..100")
        self.blueprint = build_blueprint(profile["channels"], metadata=True)
        self.web_process = None
        try:
            self._configure(profile, headless, save, connect, web, web_port, grpc_port)
        except BaseException:
            try:
                self.recording.disconnect()
            finally:
                self._stop_web()
            raise

    def _image(self, image):
        quality = getattr(self, "jpeg_quality", 0)
        if quality:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("Cannot encode Rerun preview image")
            return rr.EncodedImage(contents=encoded, media_type="image/jpeg")
        return rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def log(self, pair, stamp, results, *, inference_ran=True):
        self.recording.set_time(TIMELINE, duration=stamp)
        self.recording.set_time("frame", sequence=pair.sequence)
        for i, (image, hands) in enumerate(zip(pair.images, results)):
            root = f"/head/eye{i}"
            # RRD is a compact visual preview; LeRobot MP4 and Bayer remain
            # lossless. Raw RGB logs exhaust a 1 GiB viewer after a few seconds.
            visual = self._image(image)
            self.recording.log(f"{root}/rgb", visual)
            # Explicitly clear disappeared IDs and all previous frame annotations.
            self.recording.log(f"{root}/hands", rr.Clear(recursive=True))
            self.recording.log(f"{root}/status", rr.Points2D(
                # Rerun centers labels on the anchor, so leave space on both sides.
                [[image.shape[1] / 2, 35]], radii=0,
                labels=[f"Frame {pair.sequence} | " + (f"{len(hands)} hand(s)" if inference_ran else "not inferred")],
                colors=[240, 240, 240], show_labels=True, draw_order=30,
            ))
            for hand in hands:
                color = {"Left": [75, 210, 255], "Right": [255, 180, 60]}.get(
                    hand.handedness, [180, 180, 180])
                valid = hand.visible & np.isfinite(hand.points).all(axis=1)
                entity = f"{root}/hands/{hand.id}"
                self.recording.log(f"{entity}/points", rr.Points2D(
                    hand.points[valid], colors=color, radii=3, draw_order=20))
                lines = [hand.points[[a, c]] for a, c in CONNECTIONS if valid[a] and valid[c]]
                if lines:
                    self.recording.log(f"{entity}/bones", rr.LineStrips2D(
                        lines, colors=color, radii=1.5, draw_order=10))
                label = f"#{hand.id} {hand.handedness} | {hand.gesture}"
                if hand.score is not None:
                    label += f" {hand.score:.0%}"
                label += f" | {hand.pinch}"
                if valid.any():
                    anchor = hand.points[valid].min(axis=0) + [0, -12]
                    anchor = np.clip(anchor, [0, 0], [image.shape[1] - 1, image.shape[0] - 1])
                    self.recording.log(f"{entity}/label", rr.Points2D(
                        [anchor], radii=0, colors=color, labels=[label],
                        show_labels=True, draw_order=30))

    def log_session(self, metadata):
        metadata = {**metadata, "rerun_preview": {
            "image_encoding": "jpeg" if self.jpeg_quality else "raw_rgb",
            "jpeg_quality": self.jpeg_quality or None,
            "note": "Preview encoding only; dataset videos and raw transport retain original samples.",
        }}
        self.recording.log("/recording/session", rr.TextDocument(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str)), static=True)

    def log_frame_metadata(self, row):
        # Leave every saved field inspectable, including raw recognizer output.
        self.recording.log("/recording/frame", rr.TextDocument(
            json.dumps(row, ensure_ascii=False, indent=2, default=str)))
        metadata = row.get("observation.capture_metadata_json", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not metadata:
            return
        stamp = float(row.get("observation.capture_time", 0))
        image_time = metadata.get("image_timestamp_seconds")
        if image_time is not None and image_time < getattr(self, "_previous_device_image_time", -np.inf):
            self._last_imu_time = -np.inf
        if image_time is not None:
            self._previous_device_image_time = image_time
        for sample in metadata.get("imu_samples", []):
            device_time = sample.get("timestamp_seconds")
            if device_time is not None:
                if device_time <= getattr(self, "_last_imu_time", -np.inf):
                    continue
                self._last_imu_time = device_time
                self.recording.set_time("device_time", duration=device_time)
                # Samples use the device image-end offset anchored at host
                # receive time; this is not exposure/host clock calibration.
                self.recording.set_time(TIMELINE, duration=stamp + (device_time - image_time if image_time is not None else 0))
            for field, label in (("acceleration_m_s2", "acceleration_m_s2"),
                                 ("angular_velocity_rad_s", "angular_velocity_rad_s")):
                values = sample.get(field)
                if values is not None:
                    for axis, value in zip("xyz", values):
                        self.recording.log(f"/sensors/imu/{label}/{axis}", rr.Scalars(float(value)))
            if sample.get("temperature_c") is not None:
                self.recording.log("/sensors/imu/temperature_c", rr.Scalars(sample["temperature_c"]))
        self.recording.set_time(TIMELINE, duration=stamp)
        self.recording.disable_timeline("device_time")
