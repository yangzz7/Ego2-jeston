"""Input adapters produce complete frame pairs in configured input order."""
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import math
import time

import cv2
import numpy as np

from .config import resolve_path
from .cyper_metadata import CyperMetadataDecoder, metadata_words


@dataclass
class Pair:
    images: tuple
    stamp: float
    received: float
    sequence: int
    metadata: dict | None = None
    raw_transport: np.ndarray | None = None
    display_transport: np.ndarray | None = None


def decode_cyper(frame, width, height, bayer="auto"):
    words = metadata_words(frame, width, height)
    raw = frame.reshape(height, width, 2)
    hw, sw = int(words[0]), int(words[1])
    if hw != 2 or sw not in (3, 4, 5, 6, 7):
        raise ValueError(f"Unsupported/invalid SmartSens metadata {hw}/{sw}; this adapter expects S2 stereo, not S0 quad/ordinary YUV")
    codes = {"RG": cv2.COLOR_BayerRG2BGR, "BG": cv2.COLOR_BayerBG2BGR,
             "GR": cv2.COLOR_BayerGR2BGR, "GB": cv2.COLOR_BayerGB2BGR}
    phases = (["BG" if sw in (4, 5, 6, 7) else "RG", "RG"] if bayer == "auto" else bayer)
    if len(phases) != 2 or any(p not in codes for p in phases):
        raise ValueError("bayer must be auto or two OpenCV code names: RG/BG/GR/GB")
    images = []
    for lane in range(2):
        plane = np.ascontiguousarray(raw[:, :, lane])
        # Metadata occupies the final row; fill using the same Bayer parity.
        # Keep the original coordinate grid and mask this region in the profile.
        plane[-1] = plane[-3]
        images.append(cv2.cvtColor(plane, codes[phases[lane]]))
    return tuple(images), (hw, sw)


def white_balance(image):
    sample = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
    means = sample.mean(axis=(0, 1))
    gains = np.clip(means.mean()/np.maximum(means, 1), 0.5, 2)
    return cv2.merge([cv2.LUT(image[:, :, c], np.clip(np.arange(256)*gains[c], 0, 255).astype(np.uint8)) for c in range(3)])


def discover_cyper(serial=None):
    """Only choose this vendor's capture nodes; never fall back to a webcam."""
    candidates = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        if (node / "index").exists() and (node / "index").read_text().strip() != "0":
            continue
        dev = (node / "device").resolve()
        for parent in [dev, *dev.parents]:
            if not (parent / "idVendor").exists():
                continue
            vid = (parent / "idVendor").read_text().strip().lower()
            pid = (parent / "idProduct").read_text().strip().lower()
            actual = (parent / "serial").read_text().strip() if (parent / "serial").exists() else ""
            if (vid, pid) == ("04b4", "00f9") and (not serial or serial.lower() == actual.lower()):
                candidates.append("/dev/"+node.name)
            break
    if len(candidates) != 1:
        identity = f" for SN {serial}" if serial else ""
        raise RuntimeError(f"Expected one Cyperstereo capture node{identity}, found {candidates}; "
                           "check the camera connection and serial number")
    return candidates[0]


def split_stereo(frame, layout):
    if layout == "sbs":
        if frame.shape[1] % 2:
            raise ValueError("SBS input width must be even")
        return tuple(np.ascontiguousarray(a) for a in np.split(frame, 2, axis=1))
    if layout == "top_bottom":
        if frame.shape[0] % 2:
            raise ValueError("Top-bottom input height must be even")
        return tuple(np.ascontiguousarray(a) for a in np.split(frame, 2, axis=0))
    raise ValueError("layout must be sbs or top_bottom")


def decode_ros_image(msg):
    """Respect ROS row stride; cv_bridge is not required."""
    channels = {"bgr8": 3, "rgb8": 3, "mono8": 1, "8UC1": 1}
    if msg.encoding not in channels:
        raise ValueError(f"Unsupported ROS encoding: {msg.encoding}; publish bgr8/rgb8/mono8")
    count = channels[msg.encoding]
    if msg.step < msg.width*count:
        raise ValueError("ROS Image.step is shorter than one row")
    buf = np.frombuffer(bytes(msg.data),dtype=np.uint8)
    if buf.size < msg.height*msg.step:
        raise ValueError("Truncated ROS Image data")
    rows = buf[:msg.height*msg.step].reshape(msg.height,msg.step)
    pixels = rows[:,:msg.width*count].reshape(msg.height,msg.width,count).copy()
    if count==1:
        return cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR) if msg.encoding=="rgb8" else pixels


class TimestampMatcher:
    """Bounded, ordered timestamp pairing; unmatched older frames are dropped."""
    def __init__(self,max_skew):
        if not math.isfinite(max_skew) or max_skew<0:
            raise ValueError("max_skew_ms must be nonnegative")
        self.skew = max_skew
        self.queues = [deque(maxlen=8),deque(maxlen=8)]
        self.last = [-math.inf,-math.inf]

    def push(self,index,stamp,image):
        if not math.isfinite(stamp) or stamp<=self.last[index]:
            return None
        self.last[index] = stamp
        self.queues[index].append((stamp,image))
        a,b = self.queues
        pair = None
        while a and b:
            delta = a[0][0]-b[0][0]
            if abs(delta)<=self.skew:
                ta,ia = a.popleft()
                tb,ib = b.popleft()
                pair = ((ia,ib),(ta+tb)*0.5)
            elif delta<0:
                a.popleft()
            else:
                b.popleft()
        return pair


class RosReceiver:
    def __init__(self,cfg):
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.node import Node
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError("ROS 2 input requires rclpy/sensor_msgs in this Python environment; source the ROS setup first") from exc
        topics = cfg.get("topics",["/cam0/image_raw","/cam1/image_raw"])
        if len(topics)!=2 or topics[0]==topics[1]:
            raise ValueError("ROS input needs two distinct topics in A/B order")
        self.matcher = TimestampMatcher(float(cfg.get("max_skew_ms",1))/1000)
        self.latest = None
        self.context = Context()
        rclpy.init(args=[],context=self.context)
        self.node = Node("ego2_stereo_preview",context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.subs = [self.node.create_subscription(Image,topic,lambda msg,i=i: self.receive(i,msg),qos_profile_sensor_data)
                     for i,topic in enumerate(topics)]

    def receive(self,index,msg):
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        pair = self.matcher.push(index,stamp,decode_ros_image(msg))
        if pair is not None:
            self.latest = pair

    def read(self,stop):
        while not stop.is_set() and self.context.ok():
            self.executor.spin_once(timeout_sec=0.05)
            if self.latest is not None:
                pair,self.latest = self.latest,None
                return pair
        return None

    def close(self):
        self.executor.shutdown(timeout_sec=1)
        self.node.destroy_node()
        self.context.try_shutdown()


class Source:
    def __init__(self, profile, cameras, *, decode_images=True):
        self.profile, self.cameras = profile, cameras
        self.cfg = profile["input"]
        self.kind = self.cfg["type"]
        self.decode_images = decode_images
        if not decode_images and (self.kind != 'cyper' or not self.cfg.get('preserve_display_raw')):
            raise ValueError('RAW-only capture requires Cyperstereo and preserve_display_raw')
        self.cap = None
        self.cuda_decoder = None
        self.ros = None
        self.images = None
        self.metadata_decoder = CyperMetadataDecoder(self.cfg.get("acceleration_scale", 9.80665))
        self.count = 0
        self.next_time = time.monotonic()
        self.base = profile["_base"]
        self.fps = float(self.cfg.get("fps", 30))
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("input.fps must be positive")

    def open(self):
        c = self.cfg
        if self.kind == "images":
            self.images = tuple(cv2.imread(str(resolve_path(self.base, c[k]))) for k in ("a", "b"))
            if any(x is None for x in self.images):
                raise ValueError("Cannot read input.a/input.b images")
        elif self.kind == "demo":
            self.images = demo_images(self.cameras, self.profile["channels"], self.profile["reference"])
        elif self.kind == "ros2":
            self.ros = RosReceiver(c)
        elif self.kind in ("cyper", "video", "uvc"):
            if self.kind == "cyper":
                device = c.get("device", "auto")
                if device == "auto":
                    device = discover_cyper(c.get("serial"))
                self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            elif self.kind == "video":
                self.cap = cv2.VideoCapture(str(resolve_path(self.base, c["path"])))
            else:
                self.cap = cv2.VideoCapture(c.get("device", 0))
                if c.get("fourcc"):
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c["fourcc"]))
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open input device/video; check device, permissions and whether another program owns it")
            if self.kind != "video":
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(c.get("width", 1280)))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(c.get("height", 1024)))
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                buffers = getattr(self, 'capture_buffer_count', 1)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffers)
                if buffers != 1 and int(self.cap.get(cv2.CAP_PROP_BUFFERSIZE)) != buffers:
                    self.cap.release()
                    raise RuntimeError(f'Camera did not accept {buffers} capture buffers')
            # Retain the negotiated device/file rate, including devices that
            # cannot honor the requested rate, for dataset timing metadata.
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if math.isfinite(fps) and fps > 0 else self.fps
            if self.kind == "cyper" and self.cap.get(cv2.CAP_PROP_CONVERT_RGB) != 0:
                raise RuntimeError("Cannot disable RGB conversion on Cyperstereo transport")
            if self.kind == "cyper" and self.decode_images:
                from .cyper_cuda import capture_backend, CudaDecoder
                backend = capture_backend()
                if backend == "cuda":
                    try:
                        self.cuda_decoder = CudaDecoder(int(c.get("width",1280)), int(c.get("height",1024)))
                    except Exception:
                        self.cap.release()
                        raise
                print(f"[capture-convert] {backend}: RAW lanes + Bayer to BGR; USB/metadata on CPU", flush=True)
            elif self.kind == 'cyper':
                print('[capture-convert] RAW-only for display ISP; no duplicate Bayer conversion', flush=True)
        else:
            raise ValueError(f"Unsupported input.type={self.kind}; use cyper, images, video, uvc, ros2 or demo")
        self.next_time = time.monotonic()

    def read(self, stop):
        metadata = None
        raw_transport = None
        display_transport = None
        if self.kind in ("images", "demo", "video"):
            if stop.wait(max(0, self.next_time-time.monotonic())):
                return None
            self.next_time = max(self.next_time+1/self.fps, time.monotonic())
        if self.ros is not None:
            pair = self.ros.read(stop)
            if pair is None:
                return None
            images,stamp = pair
        elif self.images is not None:
            images = self.images
            stamp = self.count/self.fps
        else:
            for attempt in range(4):
                ok, frame = self.cap.read()
                if not ok and self.kind == "video" and self.cfg.get("loop", True):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                if not ok:
                    if self.kind == "video":
                        return None
                    raise RuntimeError("Camera read failed; no stale frame is substituted")
                host_receive_monotonic = time.monotonic()
                host_receive_wall_time_ns = time.time_ns()
                # Some Cyper UVC devices return uninitialized zero metadata at
                # stream start with multiple buffers. Only discard that exact
                # startup condition, before any valid pair/IMU can be published.
                # Runtime corruption and unsupported firmware still fail closed.
                if (self.kind != 'cyper' or self.count != 0 or
                        getattr(self, 'capture_buffer_count', 1) <= 1 or
                        np.any(metadata_words(frame, int(self.cfg.get('width',1280)),
                                               int(self.cfg.get('height',1024))))):
                    break
                print(f'[capture] skipped empty startup transport {attempt+1}/4',flush=True)
                if stop.is_set():return None
            else:
                raise RuntimeError('Camera supplied only empty startup metadata (4 frames)')
            if self.kind == "cyper":
                width, height = int(self.cfg.get("width", 1280)), int(self.cfg.get("height", 1024))
                # Preserve original metadata and optionally packed Bayer bytes
                # before demosaicing, metadata-row replacement or white balance.
                metadata = self.metadata_decoder.decode(frame, width, height)
                metadata.update({"host_receive_monotonic_seconds": host_receive_monotonic,
                                 "host_receive_wall_time_ns": host_receive_wall_time_ns,
                                 "host_timestamp_semantics": "after_capture_read_before_image_processing",
                                 "source_fps": self.fps})
                if self.cfg.get("preserve_raw", False):
                    raw_transport = frame.reshape(height, width, 2).copy()
                if self.cfg.get("preserve_display_raw", False):
                    # Separate field: --no-raw must still omit dataset RAW.
                    display_transport = (raw_transport if raw_transport is not None else
                                         frame.reshape(height, width, 2).copy())
                fused_balance = False
                if not self.decode_images:
                    images, firmware = (), self.metadata_decoder.firmware
                elif self.cuda_decoder is not None:
                    fused_balance = (getattr(self, 'fuse_white_balance', False) and
                                     self.cfg.get('white_balance', True) and (width,height)==(1280,1024))
                    if fused_balance:
                        try:
                            images, firmware = self.cuda_decoder.decode(frame, self.cfg.get("bayer", "auto"), white_balance=True)
                        except RuntimeError as exc:
                            # Retry this same frame through the existing separate
                            # stages, never substitute a previous image or metadata.
                            self.fuse_white_balance = fused_balance = False
                            print(f'[gesture-preprocess] fused CUDA unavailable; separate original stages: {exc}',flush=True)
                    if not fused_balance:
                        images, firmware = self.cuda_decoder.decode(frame, self.cfg.get("bayer", "auto"))
                else:
                    images, firmware = decode_cyper(frame, width, height, self.cfg.get("bayer", "auto"))
                if self.count == 0:
                    if fused_balance:print('[gesture-preprocess] RAW/Bayer + stereo white balance: fused CUDA; original pixels',flush=True)
                    print(f"[input] SmartSens {firmware[0]}/{firmware[1]}, A=C1(low byte), B=C2(high byte)", flush=True)
                if self.decode_images and self.cfg.get("white_balance", True) and not fused_balance:
                    balance = getattr(self, "white_balance_processor", white_balance)
                    images = tuple(balance(x) for x in images)
            else:
                images = split_stereo(frame, self.cfg.get("layout", "sbs"))
            # Paired transport has one acquisition instant. This host stamp is
            # NOT a measurement of sensor exposure time / photon-to-screen delay.
            stamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if self.kind == "video" else host_receive_monotonic
        self.count += 1
        received = time.monotonic()
        if metadata is not None:
            metadata["host_decode_complete_monotonic_seconds"] = received
        return Pair(images, stamp, received, self.count, metadata, raw_transport, display_transport)

    def close(self):
        if self.cuda_decoder is not None:
            self.cuda_decoder.close()
            self.cuda_decoder = None
        if self.ros is not None:
            self.ros.close()
        if self.cap is not None:
            self.cap.release()


def demo_images(cameras, channels, reference):
    """Render a synthetic textured plane at 2 m from actual calibrated rays.

    The two images independently render the same 3-D plane, including baseline.
    This is a geometry smoke-test, not a simulation of real lens image quality.
    """
    images = []
    root_to_ref = cameras[reference].T_from_root
    for ch in channels:
        cam = cameras[ch["camera"]]
        w, h = cam.size
        y, x = np.mgrid[:h, :w].astype(np.float64)
        yn = (y-cam.K[1, 2])/cam.K[1, 1]
        xn = (x-cam.K[0, 2]-cam.K[0, 1]*yn)/cam.K[0, 0]
        rho = np.hypot(xn, yn)
        lo, hi = np.zeros_like(rho), np.full_like(rho, cam.theta_limit)
        def radial(t):
            t2 = t*t
            a,b,c,d = cam.D
            return t*(1+t2*(a+t2*(b+t2*(c+t2*d))))
        for _ in range(35):
            mid = (lo+hi)*0.5
            select = radial(mid) < rho
            lo = np.where(select, mid, lo)
            hi = np.where(select, hi, mid)
        theta = (lo+hi)*0.5
        scale = np.sin(theta)/np.maximum(rho, 1e-12)
        rays = np.stack([xn*scale, yn*scale, np.cos(theta)], axis=-1)
        to_ref = root_to_ref @ np.linalg.inv(cam.T_from_root)
        rays = rays @ to_ref[:3,:3].T
        origin = to_ref[:3,3]
        good = (rho <= radial(cam.theta_limit)) & (rays[...,2] > 1e-4)
        distance = (2-origin[2])/np.maximum(rays[...,2], 1e-4)
        points = rays*distance[...,None]+origin
        xx, yy = points[...,0], points[...,1]
        checker = ((np.floor(xx*4)+np.floor(yy*4)).astype(np.int64) % 2).astype(bool)
        image = np.empty((h,w,3), np.uint8)
        image[:] = (65, 80, 100)
        image[checker] = (150, 175, 190)
        red = (abs(xx+0.6)<0.22) & (abs(yy)<0.4)
        blue = (xx-0.6)**2+(yy+0.2)**2 < 0.15
        image[red] = (45,45,230)
        image[blue] = (235,130,35)
        image[~good] = 0
        images.append(image)
    return tuple(images)
