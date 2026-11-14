"""Display-only RAW ISP, with a bounded queue of complete visualization frames.

The detector never receives these images. Every queued item owns a snapshot of
its overlays and poses; dropping an item drops the whole visualization frame.
"""
import copy
import threading
import time

import numpy as np

from ..common.config import resolve_path
from ..common.paths import ROOT


def configure_display_isp(cfg):
    settings = cfg["viewer"].setdefault("display_isp", {"enabled": False})
    if not isinstance(settings, dict) or not isinstance(settings.get("enabled", False), bool):
        raise ValueError("viewer.display_isp.enabled must be a boolean")
    settings.setdefault("enabled", False)
    if settings["enabled"]:
        if settings.get("config", "auto") == "auto":
            serial = cfg["input"].get("serial", "")
            path = ROOT / "cyperstereo_isp_portable" / "configs" / f"{serial}.yaml"
            settings["config"] = str(path)
            if not path.is_file():
                settings["enabled"] = False
                settings["disabled_reason"] = f"No display ISP profile for SN {serial}: {path}; showing original RGB"
                return settings
        for key in ("config", "library"):
            if not isinstance(settings.get(key), str) or not settings[key]:
                raise ValueError(f"viewer.display_isp.{key} must be a path")
            settings[key] = str(resolve_path(cfg["_base"], settings[key]))
    return settings


def display_isp_active(cfg, args):
    return (cfg["viewer"].get("display_isp", {}).get("enabled", False)
            and not args.no_viz and not args.demo and cfg["input"]["type"] == "cyper")


def create_visualizer(cfg, cameras, board, args, save, *, viewer_factory=None):
    from .visualizer import Visualizer
    if cfg["viewer"].get("display_isp", {}).get("disabled_reason"):
        print(f"[display-isp] {cfg['viewer']['display_isp']['disabled_reason']}", flush=True)
    active = display_isp_active(cfg, args)
    processor = None
    if active:
        # Namespace import from this checkout: no sys.path edits or external SDK.
        from cyperstereo_isp_portable.python.cyperstereo_isp import Isp
        settings = cfg["viewer"]["display_isp"]
        processor = Isp(config=settings["config"], library=settings["library"])
    try:
        viewer = (viewer_factory or Visualizer)(cfg, cameras, board, args, save, display_isp=active)
    except BaseException:
        if processor is not None:
            processor.close()
        raise
    if processor is None:
        return viewer
    print("[display-isp] 四画面启用 ISP；检测输入不变。独立显示线程，仅保留最新待显示帧。", flush=True)
    return IspVisualizer(viewer, processor, record_raw=save is not None and getattr(viewer, 'record_raw', True))


class IspVisualizer:
    """One processing item plus one pending item; never wait for ISP in log().

    Source gives each Pair independent image/RAW storage. We keep those buffers
    alive, and copy mutable detector state before the next pipeline iteration.
    Hidden /raw images are logged on the caller thread for EVERY inference
    frame, so display drops cannot break poses.jsonl ↔ RRD regression replay.
    Rerun timeline state is thread-local; both writers set both timelines.
    """
    def __init__(self, viewer, processor, record_raw=True):
        self.viewer, self.processor = viewer, processor
        self.record_raw = record_raw
        self.condition = threading.Condition()
        self.pending = None
        self.closing = self.closed = False
        self.error = None
        self.submitted = self.dropped = self.rendered = 0
        self.enhanced = self.fallback = 0
        self.isp_ms = self.work_ms = self.wait_ms = 0.
        self.last_frame = None
        self.warned = set()
        self.thread = threading.Thread(target=self._run, name="aprilgrid-display-isp", daemon=True)
        self.thread.start()

    def log(self, pair, stamp, observations, track, processing_ms, diagnostics=None, camera_pose=None):
        with self.condition:
            self._check_open()
        # Detection only reads Pair images; the RAW copy was made before decode.
        observations, track, diagnostics, camera_pose = copy.deepcopy(
            (observations, track, diagnostics, camera_pose))
        if self.record_raw:
            self.viewer.log_raw(pair, stamp)
        item = (pair, stamp, observations, track, processing_ms, diagnostics, camera_pose, time.perf_counter())
        with self.condition:
            self._check_open()
            self.submitted += 1
            if self.pending is not None:
                self.dropped += 1
            self.pending = item
            self.condition.notify_all()

    def _check_open(self):
        if self.error is not None:
            raise RuntimeError(f"Display worker failed: {self.error}") from self.error
        if self.closing:
            raise RuntimeError("Display worker is closed")

    def _enhance(self, pair):
        started = time.perf_counter()
        info = {"frame": pair.sequence, "source_time_s": pair.stamp, "status": "enhanced"}
        try:
            if pair.raw_transport is None:
                raise ValueError("no original Bayer transport")
            images = self.processor.process_transport(pair.raw_transport)
            if len(images) != len(pair.images) or any(
                image.dtype != np.uint8 or image.shape != original.shape
                for image, original in zip(images, pair.images)
            ):
                raise ValueError("ISP output changed image geometry or format")
            gains = self.processor.last_white_balance_rgb
            info["white_balance_rgb"] = gains.tolist() if gains is not None and np.isfinite(gains).all() else None
        except Exception as exc:
            # Never reuse the last enhanced image with this frame's overlays.
            images = pair.images
            info.update(status="fallback", reason=str(exc))
            if type(exc).__name__ not in self.warned:
                self.warned.add(type(exc).__name__)
                print(f"[display-isp] 本帧使用原图，检测继续：{exc}", flush=True)
        info["processing_ms"] = (time.perf_counter()-started)*1000
        return images, info

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.closing)
                    if self.pending is None:
                        return
                    item, self.pending = self.pending, None
                pair, stamp, obs, track, ms, diagnostics, pose, submitted_at = item
                started = time.perf_counter()
                images, info = self._enhance(pair)
                self.viewer.log(pair, stamp, obs, track, ms, diagnostics, pose,
                                display_images=images, display_info=info, record_raw=False)
                with self.condition:
                    self.rendered += 1
                    self.enhanced += info["status"] == "enhanced"
                    self.fallback += info["status"] == "fallback"
                    self.isp_ms += info["processing_ms"]
                    self.work_ms += (time.perf_counter()-started)*1000
                    self.wait_ms += (started-submitted_at)*1000
                    self.last_frame = pair.sequence
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.pending = None
                self.condition.notify_all()

    def statistics(self):
        with self.condition:
            return {"enabled": True, "submitted_frames": self.submitted,
                    "displayed_frames": self.rendered, "dropped_display_frames": self.dropped,
                    "enhanced_frames": self.enhanced, "fallback_frames": self.fallback,
                    "last_displayed_frame": self.last_frame, "max_pending_pairs": 1,
                    "isp_ms_mean": self.isp_ms/self.rendered if self.rendered else None,
                    "display_work_ms_mean": self.work_ms/self.rendered if self.rendered else None,
                    "queue_wait_ms_mean": self.wait_ms/self.rendered if self.rendered else None,
                    "raw_recorded_for_every_inference_frame": self.record_raw}

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closing = True
            self.condition.notify_all()
        # Finish the active and latest pending frames before closing either sink.
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("Display worker did not stop; ISP/sink kept alive to avoid concurrent teardown")
        self.closed = True
        try:
            self.processor.close()
        finally:
            self.viewer.close()
        if self.error is not None:
            raise RuntimeError(f"Display worker failed: {self.error}") from self.error
