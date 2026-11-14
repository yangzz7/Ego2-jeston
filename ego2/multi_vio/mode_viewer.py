"""One Rerun owner; bounded complete-frame previews from local child processes.

Archetypes are pickled only across our inherited multiprocessing queues, never
from a socket or file supplied by another process/user. Recording of sensor and
algorithm data stays in each worker and is independent of this lossy preview.
"""
import json
from concurrent.futures import Future, ThreadPoolExecutor
import io
import pickle
import queue
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as b

from ..common.viewer import ViewerConnection, TIMELINE
from ..gesture.visualizer import Visualizer as GestureVisualizer
from ..aprilgrid.visualizer import Visualizer as AprilGridVisualizer
from .roles import VIEWER_LABELS
from .preview_transport import PreviewImage, encode_image, pack_images


def _restore_arrow_array(data):
    with pa.ipc.open_stream(data) as reader:
        return reader.read_next_batch().column(0)


def _encode_aprilgrid_original(image, quality):
    # Preserve the existing recording/fallback JPEG implementation exactly;
    # only move this work out of the detection process.
    return rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=quality)


class _PreviewPickler(pickle.Pickler):
    def reducer_override(self, value):
        if isinstance(value, pa.Array):
            # PyArrow's default pickle reconstruction loses non-nullable
            # children of fixed-size lists. Rerun requires the exact schema
            # for positions/lines and silently skips those changed types.
            batch = pa.RecordBatch.from_arrays([value], names=['value'])
            buffer = pa.BufferOutputStream()
            with pa.ipc.new_stream(buffer, batch.schema) as writer:
                writer.write_batch(batch)
            return _restore_arrow_array, (buffer.getvalue().to_pybytes(),)
        return NotImplemented


def build_blueprint(cfg, *, imu=True):
    if cfg['mode'] == 'reconstruction':
        from ..reconstruction.visualizer import build_blueprint as reconstruction_blueprint
        return reconstruction_blueprint()
    rows = []
    for role in cfg['viewer']['row_order']:
        sn = cfg['devices'][role]['sn'] or 'DEMO'
        views = [
            b.Spatial2DView(name=f'{VIEWER_LABELS[role]} · {sn} | {eye} / {cam}',
                            origin=f'/cameras/{role}/{cam}', contents='$origin/**')
            for eye, cam in (('Left', 'cam1'), ('Right', 'cam0'))]
        if cfg['mode'] == 'gesture' and imu:
            from .orientation_preview import charts
            plots = charts(role, f'{VIEWER_LABELS[role]} · {sn}')
            if len(cfg['viewer']['row_order']) == 1:
                rows.extend([b.Horizontal(*views, column_shares=[1, 1]),
                             plots])
            else:
                rows.append(b.Horizontal(*views, plots,
                                         column_shares=[1, 1, 3]))
        else:
            rows.append(b.Horizontal(*views, column_shares=[1, 1]))
    shares = [1.25, 1] if cfg['mode'] == 'gesture' and imu and len(cfg['viewer']['row_order']) == 1 else [1]*len(rows)
    # Keep all four quaternion legend entries visible on a 1080p display.
    # Collapsing the event list preserves the timeline and playback controls.
    compact_time = cfg['mode'] == 'gesture' and imu and len(cfg['viewer']['row_order']) >= 3
    return b.Blueprint(b.Vertical(*rows, row_shares=shares),
                      b.TimePanel(state='collapsed' if compact_time else 'expanded', timeline=TIMELINE, play_state='following'),
                      b.BlueprintPanel(state='hidden'), b.SelectionPanel(state='hidden'),
                      auto_layout=False, auto_views=False)


class UnifiedViewer(ViewerConnection):
    def __init__(self, cfg, args, save=None):
        self.cfg = cfg
        self.recording = rr.RecordingStream('Ego2 · '+cfg['mode'])
        self.blueprint = build_blueprint(cfg, imu=save is None)
        self.web_process = None
        self.display_processors = {}
        self.file_viewer = None
        self.encoder_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="preview-jpeg")
        try:
            from ..common.display_isp import create_display_isp, live_display
            # File replay keeps the original batches, even when live preview
            # drops frames or replaces RGB with ISP output.
            if save and live_display(args):
                options = SimpleNamespace(**{**vars(args), 'headless':True, 'connect':None, 'web':False})
                self.file_viewer = UnifiedViewer(cfg, options, save)
                save = None
            self._configure(cfg, args.headless, save, args.connect, args.web, args.web_port, args.grpc_port)
            self.imu_logger = None
            self.raw_imu_logger = None
            self.body_preview = None
            self.preview_origin = None
            self.body_errors = 0
            self.last_body_tick = -float("inf")
            if cfg['mode'] == 'gesture':
                from .orientation_preview import QuaternionLogger
                self.imu_logger = QuaternionLogger(self.recording, cfg['viewer']['row_order'])
                if save is None:
                    from .raw_imu import RawImuLogger
                    from .body_preview import BodyPreview
                    self.raw_imu_logger = RawImuLogger(self.recording, cfg['viewer']['row_order'])
                    try:
                        from .hand_diagnostics import HandDiagnostics
                        diagnostics = None
                        try:
                            diagnostics = HandDiagnostics()
                            print(f'[hand-model] diagnostics: {diagnostics.path} (session {diagnostics.session})', flush=True)
                        except OSError as exc:
                            print(f'[hand-model] diagnostic file unavailable: {exc}', flush=True)
                        try:
                            self.body_preview = BodyPreview(self.recording, cfg['viewer']['row_order'], diagnostics=diagnostics)
                            if diagnostics:
                                diagnostics.emit(dict(kind='configuration',
                                    cameras={r:d.get('sn') for r,d in cfg['devices'].items()},
                                    selection=self.body_preview.settings['hand_selection'],
                                    smoothing=self.body_preview.settings['hand_smoothing']))
                        except BaseException:
                            if diagnostics: diagnostics.close()
                            raise
                    except Exception as exc:
                        print(f'[body-preview] initialization failed: {exc}; camera/recording continue', flush=True)
            self.recording.log('/session', rr.TextDocument(json.dumps(cfg, ensure_ascii=False)), static=True)
            for role, device in cfg['devices'].items():
                self.display_processors[role] = create_display_isp(cfg, device['sn'], args)
        except BaseException:
            self.close()
            raise

    def consume(self, blob, *, cutoff=None, end=None):
        batch = blob if isinstance(blob, dict) else pickle.loads(blob)
        if 'recording_stopped' in batch:
            return batch
        if cutoff is not None and (batch.get('received') is None or batch['received'] < cutoff):
            return batch
        if end is not None and batch.get('received') is not None and batch['received'] >= end:
            return batch
        if getattr(self, 'file_viewer', None):
            self.file_viewer.consume(blob, cutoff=cutoff, end=end)
        if batch.get('hand_visual') and getattr(self, 'body_preview', None):
            now = time.monotonic()-self.preview_origin if self.preview_origin is not None else batch['hand_visual']['received']
            self._body_call('consume_visual', batch['hand_visual'], now)
        processor = getattr(self, 'display_processors', {}).get(batch['role'])
        if processor and 'display_raw' in batch:
            processor.submit(batch['display_raw'], None, lambda images: self._log_batch(batch, images, cutoff=cutoff),
                             owned=batch.get('owned_preview_buffers', False))
        else:
            self._log_batch(batch, cutoff=cutoff)
        return batch

    def consume_imu(self, packet):
        if self.imu_logger is not None:
            self.imu_logger.consume(packet)
        if getattr(self, 'raw_imu_logger', None) is not None and len(packet.get('raw_imu', [])):
            self.raw_imu_logger.consume(dict(role=packet['role'], imu=packet['raw_imu'],
                clock=packet.get('clock', {}), raw_diagnostics=packet.get('raw_diagnostics', {})))
        if getattr(self, 'body_preview', None):
            now = time.monotonic()-self.preview_origin if self.preview_origin is not None else float(packet['imu'][-1,0])
            self._body_call('consume_imu', packet, now)

    def _body_call(self, method, *args):
        try:
            getattr(self.body_preview, method)(*args)
        except Exception as exc:
            self.body_errors += 1
            print(f'[body-preview] {method}: {type(exc).__name__}: {exc}', flush=True)
            if self.body_errors >= 3:
                print('[body-preview] disabled after repeated errors; camera/recording continue', flush=True)
                self.body_preview.close()
                self.body_preview = None

    def tick(self, now):
        if getattr(self, 'body_preview', None) and now-self.last_body_tick >= .05:
            self.last_body_tick = now
            self._body_call('tick', now)

    def _log_batch(self, batch, images=None, *, cutoff=None):
        replacements = {} if images is None else {
            f'/cameras/{batch["role"]}/{cam}/{suffix}': image
            for cam, image in zip(batch.get('display_eyes',('cam0','cam1')), images)
            for suffix in ('rgb','image')}
        # Capture pixels and metadata from this immutable batch. Encoding may
        # run concurrently across eyes; no image or overlay is published until
        # every encoding completes. The two-worker pool is shared by all roles.
        pool = getattr(self, 'encoder_pool', None)
        def encode(image, quality, rerun_compress=False):
            encoder = _encode_aprilgrid_original if rerun_compress else encode_image
            return pool.submit(encoder, image, quality) if pool else encoder(image, quality)
        prepared = []
        for path, archetypes, kwargs, times in batch['logs']:
            if path in replacements:
                values = [encode(replacements[path], self.cfg['viewer'].get('jpeg_quality',90))]
            else:
                values = [encode(v.pixels, v.quality, v.rerun_compress) if isinstance(v, PreviewImage) else v for v in archetypes]
            prepared.append((path, values, kwargs, times))
        resolved = [(path, tuple(v.result() if isinstance(v, Future) else v for v in values), kwargs, times)
                    for path, values, kwargs, times in prepared]
        for path, archetypes, kwargs, times in resolved:
            # Reset timelines between devices and between IMU/image log items.
            self.recording.reset_time()
            for name, value in times.items():
                if cutoff is not None and name == TIMELINE:
                    value = dict(duration=max(0.,batch['received']-cutoff))
                self.recording.set_time(name, **value)
            self.recording.log(path, *archetypes, **kwargs)

    def close(self):
        if getattr(self, 'body_preview', None): self.body_preview.close()
        for processor in getattr(self, 'display_processors', {}).values():
            if processor: processor.close()
        if getattr(self, 'encoder_pool', None):
            self.encoder_pool.shutdown(wait=True)
            self.encoder_pool = None
        if getattr(self, 'file_viewer', None):
            self.file_viewer.close()
            self.file_viewer = None
        super().close()

    def state(self, role, stamp, text):
        if getattr(self, 'file_viewer', None): self.file_viewer.state(role,stamp,text)
        self.recording.reset_time()
        self.recording.set_time(TIMELINE, duration=max(0., stamp))
        for cam in ('cam1', 'cam0'):
            self.recording.log(f'/cameras/{role}/{cam}/worker_state',
                rr.Points2D([[640, 95]], labels=[text], show_labels=True, radii=0,
                            colors=[255, 190, 70], draw_order=100))


class BatchRecording:
    def __init__(self, preview, role, origin, *, demo=False, eye_names=(), reliable=False, control=None, transport=None, imu_queue=None, attitude_alive=None):
        self.preview, self.role, self.origin = preview, role, origin
        self.demo, self.eye_names = demo, tuple(eye_names)
        self.reliable = reliable
        self.local = threading.local()
        self.sent = self.dropped = 0
        self.lock = threading.Lock()
        self.control = control
        if transport is not None and reliable:
            raise ValueError('Shared preview transport must never carry a durable recording')
        self.transport = transport
        self.imu_queue = imu_queue
        self.attitude_alive = attitude_alive
        self.imu_stats = {}

    def begin(self, stamp=0., *, received=None, sequence=None):
        if self.control: self.control.poll()
        self.local.received = received
        shared = stamp if self.demo else (received if received is not None else time.monotonic()) - self.origin
        self.local.offset = max(0., shared) - stamp
        self.local.times = {TIMELINE: dict(duration=max(0., shared))}
        if sequence is not None:
            self.local.times[self.role+'_frame'] = dict(sequence=int(sequence))
        self.local.logs = []
        self.local.display_raw = None
        self.local.hand_visual = None

    def display(self, pair):
        self.local.display_raw = getattr(pair, 'display_transport', None)

    def recording_end(self, token):
        # FIFO barrier: parent drains all prior batches before closing the RRD.
        self.preview.put(pickle.dumps(dict(role=self.role, recording_stopped=token)))

    def set_time(self, name, **value):
        if name == TIMELINE and 'duration' in value:
            value = dict(duration=max(0., float(value['duration']) + self.local.offset))
        elif name != TIMELINE:
            name = self.role+'_'+name
        self.local.times[name] = value

    def disable_timeline(self, name):
        self.local.times.pop(self.role+'_'+name, None)

    def path(self, path):
        for eye, cam in enumerate(self.eye_names):
            prefix = f'/head/eye{eye}'
            if path == prefix or path.startswith(prefix+'/'):
                return f'/cameras/{self.role}/{cam}'+path[len(prefix):]
        for cam in ('cam0', 'cam1'):
            prefix = '/tracking/'+cam
            if path.startswith(prefix+'/'):
                return f'/cameras/{self.role}/{cam}'+path[len(prefix):]
        return f'/diagnostics/{self.role}/'+path.lstrip('/')

    def log(self, path, *archetypes, **kwargs):
        # One visible RGB copy, no duplicate raw/display panes or hidden images.
        if path.startswith(('/raw/', '/display/')):
            return
        self.local.logs.append((self.path(path), archetypes, kwargs, dict(self.local.times)))

    def finish(self):
        logs = self.local.logs
        if not logs:
            return
        buffer = io.BytesIO()
        batch = dict(role=self.role, logs=logs, received=self.local.received)
        if self.local.hand_visual is not None:
            batch['hand_visual'] = self.local.hand_visual
        if self.local.display_raw is not None:
            batch['display_raw'] = self.local.display_raw
            batch['display_eyes'] = self.eye_names
        if self.transport is not None:
            batch = pack_images(batch, self.transport)
            if batch is None:
                self.dropped += 1
                self.local.logs = []
                return
        _PreviewPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(batch)
        blob = buffer.getvalue()
        # Serialize before queueing so errors cannot disappear in a feeder thread.
        with self.lock:
            try:
                if self.reliable:
                    self.preview.put(blob)
                else:
                    self.preview.put_nowait(blob)
                self.sent += 1
            except queue.Full:
                self.dropped += 1
        self.local.logs = []


class GestureSink(GestureVisualizer):
    def __init__(self, profile, recording, **kwargs):
        self.recording = recording
        self.jpeg_quality = min(90, int(profile['viewer'].get('jpeg_quality', 85)))
        self.imu_publisher = None
        if recording.imu_queue is not None:
            if getattr(recording, 'attitude_alive', None) is not None:
                from .orientation_service import AttitudeInput
                self.imu_publisher = AttitudeInput(recording.imu_queue, recording.role, recording.origin, recording.attitude_alive)
            else:
                from .orientation_preview import QuaternionPublisher
                self.imu_publisher = QuaternionPublisher(recording.imu_queue, recording.role, recording.origin, profile)
            recording.imu_stats = self.imu_publisher.stats

    def capture_imu(self, pair):
        if self.imu_publisher is not None:
            self.imu_publisher.capture(pair)

    def _image(self, image):
        if self.recording.transport is not None:
            return PreviewImage(image, self.jpeg_quality)
        return super()._image(image)

    def log_observations(self, pair, stamp, results, observations):
        self.log(pair, stamp, results, _observations=observations)

    def set_tracking_diagnostics(self, rows):
        self._tracking_diagnostics = {eye:row for eye,row in zip(self.recording.eye_names, rows)}

    def log(self, pair, stamp, results, **kwargs):
        observations = kwargs.pop('_observations', None)
        tracking = getattr(self, '_tracking_diagnostics', {})
        self._tracking_diagnostics = {}
        self.recording.begin(stamp, received=pair.received, sequence=pair.sequence)
        self.recording.display(pair)
        super().log(pair, stamp, results, **kwargs)
        self.recording.log('/source', rr.TextDocument(json.dumps(dict(
            source_timestamp=pair.stamp, source_sequence=pair.sequence,
            host_received=pair.received))))
        if observations is not None:
            from .hand_shape import pack_observations
            try:
                self.recording.local.hand_visual = dict(role=self.recording.role,
                    received=max(0., pair.received-self.recording.origin),
                    image_device_time=(pair.metadata or {}).get('image_timestamp_seconds'),
                    tracking=tracking,
                    observations=pack_observations(observations, results, self.recording.eye_names))
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                print(f'[body-preview] observation skipped: {exc}', flush=True)
        self.recording.finish()

    def log_session(self, metadata):
        self.recording.begin()
        super().log_session(metadata)
        self.recording.finish()

    def log_frame_metadata(self, row):
        # Full-rate IMU, capture metadata and inference results are in LeRobot.
        # The unified RRD is a bounded preview, not a second sensor archive.
        pass

    def close(self):
        if self.imu_publisher is not None:
            self.imu_publisher.flush()


class AprilGridSink(AprilGridVisualizer):
    record_raw = False

    def __init__(self, cfg, cameras, board, args, save, *, recording, **kwargs):
        self.cfg, self.cameras, self.board = cfg, cameras, board
        self.recording = recording
        root = np.linalg.inv(cameras[cfg['reference']].T_from_root)
        self.transforms = {name: camera.T_from_root @ root for name, camera in cameras.items()}

    def _image(self, image):
        # Send the original pixels in this complete batch. The parent encodes
        # either these pixels (recording/fallback) or same-frame ISP output.
        # In particular, do not JPEG-encode an image the live ISP will replace.
        return PreviewImage(image, int(self.cfg['viewer']['jpeg_quality']), rerun_compress=True)

    def log(self, pair, stamp, observations, track, processing_ms, diagnostics=None, camera_pose=None, **kwargs):
        self.recording.begin(stamp, received=pair.received, sequence=pair.sequence)
        self.recording.display(pair)
        kwargs['record_raw'] = False
        super().log(pair, stamp, observations, track, processing_ms, diagnostics, camera_pose, **kwargs)
        if camera_pose is not None:
            for cam in ('cam0', 'cam1'):
                value = camera_pose['cameras'][cam]  # Never present a held pose as current.
                if value:
                    p = ', '.join(f'{x:+.3f}' for x in value['position_m'])
                    r = ', '.join(f'{x:+.1f}' for x in value['rpy_deg'])
                    text = f'Board pose | p/m [{p}] | RPY/deg [{r}]'
                else:
                    text = 'Board pose: '+camera_pose['status'].upper()
                self.recording.log(f'/tracking/{cam}/pose_status', rr.Points2D([[640, 65]],
                    labels=[text], show_labels=True, radii=0, colors=[225, 225, 225], draw_order=40))
        self.recording.log('/source', rr.TextDocument(json.dumps(dict(
            source_timestamp=pair.stamp, source_sequence=pair.sequence, host_received=pair.received))))
        self.recording.finish()

    def close(self):
        pass
