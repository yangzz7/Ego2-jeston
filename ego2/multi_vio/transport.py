"""One native process per device; same-frame poses and lossless image rings."""
from collections import deque
import fcntl
import json
import mmap
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
import numpy as np
import zmq
from ..common.config import se3
from ..common.paths import BUILD, ROOT
from ..reconstruction.build import build_native
from ..reconstruction.calibration import write_orb_settings

IMAGE_BYTES = 1280*1024*3
SLOT_BYTES = 8+2*IMAGE_BYTES
SLOTS = 3
RAW_BYTES = 1280*1024*2
PREVIEW_SLOT_BYTES = SLOT_BYTES+8+RAW_BYTES


def check_devices(devices):
    """Read-only discovery of every requested SN before opening any device."""
    from ..common.sources import discover_cyper
    found, errors = {}, []
    for role, data in devices.items():
        sn = data['native']['serial']
        try:
            found[role] = discover_cyper(sn)
        except RuntimeError as exc:
            errors.append(f'{role} ({sn}): {exc}')
    if errors:
        raise RuntimeError('Camera discovery failed; no VIO processes started:\n'+'\n'.join(errors))
    if len(set(found.values())) != len(found):
        raise RuntimeError('Different roles resolved to the same video node; check device serial numbers')
    for role, node in found.items():
        print(f'[device] {role}: {devices[role]["native"]["serial"]} -> {node}', flush=True)
    return found


class ClockMapper:
    """Lower arrival-delay envelope; approximate host clock, not exposure sync.

    Offset follows device drift over a rolling window. Limit corrections to
    1 ms/frame and preserve monotonicity. The unknown USB latency stays unknown.
    """
    def __init__(self, origin):
        self.origin, self.offset, self.last_device, self.last_output = origin, None, None, None
        self.offsets = deque(maxlen=300)

    def update(self, device, received):
        if not np.isfinite([device, received]).all() or (self.last_device is not None and device <= self.last_device):
            raise ValueError('Non-monotonic or invalid device clock; restart the session')
        self.offsets.append(received-device)
        target = min(self.offsets)
        self.offset = target if self.offset is None else self.offset+float(np.clip(target-self.offset,-.001,.001))
        stamp = device+self.offset-self.origin
        if self.last_output is not None:
            stamp = max(stamp, self.last_output+1e-6)
        self.last_device, self.last_output = device, stamp
        return stamp, float(np.percentile(self.offsets,95)-min(self.offsets))


class RingReader:
    def __init__(self, path, version=1):
        if type(version) is not int or version not in (1, 2):
            raise ValueError('Unsupported native frame ring version')
        self.version = version
        self.stride = SLOT_BYTES if version == 1 else PREVIEW_SLOT_BYTES
        self.file = Path(path).open('rb')
        if os.fstat(self.file.fileno()).st_size != SLOTS*self.stride:
            self.file.close()
            raise ValueError('Invalid native frame ring size')
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def read(self, packet):
        bundle = self.read_bundle(packet)
        return None if bundle is None else bundle[0]

    def read_bundle(self, packet):
        slot, sequence = packet['slot'], packet['frame_id']
        if type(slot) is not int or not 0 <= slot < SLOTS or type(sequence) is not int or sequence < 0:
            raise ValueError('Invalid native frame ring notification')
        offset = slot*self.stride
        fcntl.flock(self.file.fileno(), fcntl.LOCK_SH)
        try:
            if struct.unpack_from('<Q',self.mapping,offset)[0] != sequence+1:
                return None
            images = np.ndarray((2,1024,1280,3),dtype=np.uint8,buffer=self.mapping,offset=offset+8).copy()
            raw = None
            if self.version == 2:
                valid = struct.unpack_from('<Q', self.mapping, offset+SLOT_BYTES)[0]
                if valid not in (0, 1):
                    raise ValueError('Invalid native RAW presence flag')
                if valid:
                    raw = np.ndarray((1024,1280,2),dtype=np.uint8,
                        buffer=self.mapping,offset=offset+SLOT_BYTES+8).copy()
            return (images[0], images[1]), raw
        finally:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)

    def close(self):
        self.mapping.close()
        self.file.close()


def ensure_native(cfg, jobs=2, rebuild=False):
    executable = BUILD/'bin/ego2_multi_vio_native'
    roots = [ROOT/'ego2/reconstruction/native', ROOT/'ego2/multi_vio/native', ROOT/'ego2/common/native', Path(cfg['orb_source'])/'include',
             Path(cfg['orb_source'])/'src', Path(cfg['orb_source'])/'Thirdparty/usb/uvc']
    modified = not executable.exists() or any(p.stat().st_mtime > executable.stat().st_mtime
                 for root in roots for p in root.rglob('*') if p.is_file())
    if rebuild or modified:
        build_native(cfg, jobs)
    return executable


class Backend:
    def __init__(self, role, cfg, root, context, origin, executable, archive=None, archive_control=None, display_raw=False):
        self.role, self.cfg = role, cfg
        self.directory = Path(root)/role
        self.directory.mkdir()
        self.endpoint = 'ipc://'+str(self.directory/'stream.ipc')
        shift = write_orb_settings(cfg, self.directory/'orb.yaml')
        self.log = (self.directory/'native.log').open('wb')
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE,b'')
        self.socket.setsockopt(zmq.RCVHWM,3)
        self.socket.setsockopt(zmq.LINGER,0)
        self.socket.connect(self.endpoint)
        self.reader = None
        self.display_snapshot = None
        self.display_raw = display_raw
        self.clock = ClockMapper(origin)
        self.last_frame = -1
        self.overwritten = 0
        self.failed = False
        environment = {**os.environ, 'OMP_NUM_THREADS':'1'}
        environment.pop('EGO2_DISPLAY_RAW_RING', None)
        environment.pop('EGO2_VIO_PREVIEW_RAW', None)
        if display_raw:
            environment['EGO2_VIO_PREVIEW_RAW'] = '1'
            print(f'[vio-preview] {role}: RAW/RGB 同帧快照；增强失败仅跳过预览帧。', flush=True)
        try:
            self.process = subprocess.Popen([str(executable),cfg['vocabulary'],str(self.directory/'orb.yaml'),
                str(self.directory),self.endpoint,cfg['serial'],str(cfg['fps']),str(shift),
                str(cfg['acceleration_scale']),str(cfg['max_imu_gap']),'0','vio'] +
                ([str(archive) if archive else '',str(archive_control)] if archive_control else [str(archive)] if archive else []),
                stdout=self.log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
        except BaseException:
            self.socket.close(); self.log.close()
            raise

    def receive(self, need_images=True):
        self.display_snapshot = None
        packet = None
        # Bound draining so an active producer cannot starve its peer.
        for _ in range(8):
            try:
                candidate = self.socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if candidate.get('serial') != self.cfg['serial'] or candidate.get('protocol') != 1 or candidate.get('cameras') != ['cam0','cam1']:
                raise ValueError('Native camera identity/protocol mismatch')
            if candidate['frame_id'] <= self.last_frame:
                continue
            self.last_frame = candidate['frame_id']
            candidate['capture_time'], candidate['arrival_delay_spread_s'] = self.clock.update(candidate['timestamp'],candidate['host_monotonic'])
            if candidate['valid']:
                se3(candidate['T_world_camera'])
            packet = candidate
        if packet is None:
            return None
        images = None
        if need_images:
            if self.reader is None:
                self.reader = RingReader(self.directory/'frames.bin', 2 if self.display_raw else 1)
            bundle = self.reader.read_bundle(packet)
            if bundle is not None:
                images, raw = bundle
                self.display_snapshot = packet['frame_id'], raw
            self.overwritten += images is None
        return packet, images

    def read_display_raw(self, packet):
        # Already copied under the same lock as RGB; never consult a live slot
        # later or substitute RAW from a different frame.
        snapshot = self.display_snapshot
        if self.display_raw and snapshot is not None and snapshot[0] == packet['frame_id']:
            return snapshot[1]
        return None

    def error_tail(self):
        self.log.flush()
        return '\n'.join((self.directory/'native.log').read_text(errors='replace').splitlines()[-16:])

    def request_stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)

    def close(self):
        self.request_stop()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill(); self.process.wait()
            print(f'[warning] {self.role}: forced native shutdown after timeout',flush=True)
        if self.reader:
            self.reader.close()
        self.socket.close()
        self.log.close()
