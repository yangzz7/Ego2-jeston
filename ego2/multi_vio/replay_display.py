"""Replay-only presentation. Recorded sensor pixels and inference stay immutable."""
import json
import math
from pathlib import Path

import numpy as np

from ..common.display_isp import settings_for


class RawDisplay:
    """Read the selected source frame's RAW, in order and with bounded memory."""
    def __init__(self, root, role, profile, settings):
        from cyperstereo_isp_portable.python.cyperstereo_isp import Isp
        self.root = (Path(root)/'sensors'/role).resolve()
        self.cameras = tuple(channel['camera'] for channel in profile['channels'])
        if len(self.cameras) != 2 or set(self.cameras) != {'cam0', 'cam1'}:
            raise ValueError('Unknown RAW stereo channel order')
        self.processor = self.journal = self.file = None
        self.current = self.path = self.cached_key = self.cached_images = None
        try:
            self.journal = (self.root/'raw/capture.jsonl').open()
            self.processor = Isp(config=settings['config'], library=settings['library'])
        except BaseException:
            self.close()
            raise

    def render(self, row, prefix, originals):
        sequence = row[prefix+'source_sequence']
        key = (sequence, row[prefix+'source_timestamp'], row[prefix+'receive_timestamp'])
        if key == self.cached_key:
            return self.cached_images
        if self.current is not None and sequence < self.current['sequence']:
            raise ValueError('RAW replay source sequence went backwards')
        while self.current is None or self.current['sequence'] < sequence:
            line = self.journal.readline()
            if not line:
                raise ValueError(f'No saved RAW for source frame {sequence}')
            self.current = json.loads(line)
        item = self.current
        if item['sequence'] != sequence:
            raise ValueError(f'RAW index does not contain source frame {sequence}')
        for field in ('source_timestamp', 'receive_timestamp'):
            if not math.isclose(item[field], row[prefix+field], rel_tol=0, abs_tol=1e-6):
                raise ValueError('RAW/image timestamp mismatch')
        relative = item['raw_transport_path']
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError('Missing or invalid RAW payload path')
        target = (self.root/relative).resolve()
        if not target.is_relative_to(self.root/'raw/transport'):
            raise ValueError('RAW payload escapes its recording directory')
        descriptor = item['metadata']['raw_transport']
        shape = descriptor['shape']
        if (shape != [1024,1280,2] or np.dtype(descriptor['dtype']) != np.dtype('uint8')
                or descriptor.get('order') != 'C' or descriptor.get('encoding') != 'raw'):
            raise ValueError('Unsupported Cyper RAW geometry/encoding')
        offset, length = item['raw_transport_offset'], item['raw_transport_length']
        if type(offset) is not int or offset < 0 or type(length) is not int or length != 1024*1280*2:
            raise ValueError('Invalid RAW byte range')
        if target != self.path:
            if self.file is not None:
                self.file.close()
            self.file, self.path = target.open('rb'), target
        self.file.seek(offset)
        payload = self.file.read(length)
        if len(payload) != length:
            raise ValueError('Saved RAW payload is truncated')
        raw = np.frombuffer(payload,dtype=np.uint8).reshape(shape)
        enhanced = self.processor.process_transport(raw)
        if len(enhanced) != 2 or any(image.dtype != np.uint8 or image.shape != originals[camera].shape
                                  for camera,image in zip(self.cameras,enhanced)):
            raise ValueError('ISP changed image geometry/format')
        self.cached_key, self.cached_images = key, dict(zip(self.cameras,enhanced))
        return self.cached_images

    def close(self):
        try:
            if self.processor is not None:
                self.processor.close()
        finally:
            self.processor = None
            for name in ('file','journal'):
                handle = getattr(self,name,None)
                if handle is not None:
                    handle.close()
                    setattr(self,name,None)


class ReplayDisplay:
    def __init__(self, reader):
        self.streams = {}
        if reader.meta['mode'] != 'gesture':
            return
        for role in reader.roles:
            profile = reader.meta.get('profiles',{}).get(role,{})
            if profile.get('input',{}).get('type') != 'cyper':
                continue
            try:
                settings = settings_for(reader.meta['config'],reader.meta['devices'][role])
                if settings is None:
                    continue
                self.streams[role] = RawDisplay(reader.root,role,profile,settings)
                print(f'[replay-isp] {role}: 使用保存的 RAW 增强回放；原始数据不变。',flush=True)
            except Exception as exc:
                self.warn(role,exc)

    @staticmethod
    def warn(role, exc):
        print(f'[replay-isp] {role}: 无法继续增强，使用原始视频：{exc}',flush=True)

    def images(self, role, row, originals):
        prefix = f'observation.{role}.'
        stream = self.streams.get(role)
        if stream is None or not row[prefix+'image_valid']:
            return originals
        try:
            return stream.render(row,prefix,originals)
        except Exception as exc:
            # One warning and a stable fallback, avoiding alternating colors.
            self.streams.pop(role)
            stream.close()
            self.warn(role,exc)
            return originals

    def close(self):
        try:
            for stream in self.streams.values():
                stream.close()
        finally:
            self.streams.clear()


class ReplayFrames:
    """Keep image and annotations on the same recorded inference frame.

    Rerun's latest-at query holds the complete pair between updates. No previous
    skeleton is ever attached to a newer image. Explicit missing-camera frames
    and clock/epoch resets clear that state, and initial frames remain visible.
    """
    def __init__(self):
        self.previous = self.saved = None

    def select(self, row, prefix):
        current = (row[prefix+'epoch'],row[prefix+'source_sequence'],
                   row[prefix+'receive_timestamp'],row[prefix+'source_timestamp'])
        if (not row[prefix+'image_valid'] or (self.previous is not None and
                (current[0] != self.previous[0] or any(a < b for a,b in zip(current[1:],self.previous[1:]))))):
            self.saved = None
        self.previous = current
        if not row[prefix+'image_valid']:
            return True,None
        if row[prefix+'inference_valid']:
            # A no-hands result updates its image and clears the old skeleton.
            self.saved = current
            return True,None
        if self.saved is not None:
            age_ms = max(0.,(current[2]-self.saved[2])*1000)
            return False,f'Showing paired frame {self.saved[1]} | between recorded inference frames ({age_ms:.0f} ms)'
        return True,None
