"""Read closed RAW journals directly for a single final video encode."""
import hashlib
import json
from pathlib import Path

import numpy as np

from ..common.sources import Pair, decode_cyper, white_balance
from .dataset import GestureDataset
from .recovery import _validate_journals, _inside, _rows
from .replay import DatasetReplay, _hands


class RawGestureReplay:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.metadata = json.loads((self.root / 'meta/ego2.json').read_text())
        marker = self.metadata.get('deferred_video', {})
        if (marker.get('version') != 1 or marker.get('materialized') or
                self.metadata.get('recording_status') != 'raw_ready'):
            raise ValueError('RAW recording has not closed successfully; cache retained')
        self.profile = self.metadata['profile']
        self.cfg = self.profile['input']
        self.fps = float(self.metadata['fps'])
        if not np.isfinite(self.fps) or self.fps <= 0 or self.cfg.get('type') != 'cyper':
            raise ValueError('Invalid closed Cyper RAW profile')
        cameras = [ch['camera'] for ch in self.profile['channels']]
        if len(cameras) != 2 or set(cameras) != {'cam0', 'cam1'}:
            raise ValueError('RAW replay requires the two named stereo channels')
        self.counts = _validate_journals(self.root)
        self.total_frames = self.counts['capture_count']
        self.total_episodes = self.counts['episode_count']

    def frames(self):
        results = iter(_rows(self.root / 'raw/inference.jsonl'))
        annotation = next(results, None)
        h, w = int(self.cfg.get('height', 1024)), int(self.cfg.get('width', 1280))
        handle = path = None
        try:
            for capture in _rows(self.root / 'raw/capture.jsonl'):
                target = _inside(self.root, capture['raw_transport_path'])
                if target != path:
                    if handle is not None:
                        handle.close()
                    handle, path = target.open('rb'), target
                descriptor = capture['metadata']['raw_transport']
                if descriptor['shape'] != [h, w, 2] or np.dtype(descriptor['dtype']) != np.dtype('uint8'):
                    raise ValueError('RAW shape/dtype disagrees with the saved profile')
                handle.seek(capture['raw_transport_offset'])
                payload = handle.read(capture['raw_transport_length'])
                if len(payload) != h * w * 2:
                    raise ValueError('RAW changed or was truncated during saving')
                raw = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 2)
                images, _ = decode_cyper(raw, w, h, self.cfg.get('bayer', 'auto'))
                if self.cfg.get('white_balance', True):
                    images = tuple(white_balance(im) for im in images)
                hashes = capture['metadata'].get('deferred_rgb_sha256')
                if not isinstance(hashes, list) or len(hashes) != 2:
                    raise ValueError('Missing original RGB checksums')
                for eye, image in enumerate(images):
                    if hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest() != hashes[eye]:
                        raise ValueError(f'RAW reconstruction differs from original RGB at pair {capture["sequence"]}, eye {eye}')
                result = None
                if annotation is not None and annotation['sequence'] == capture['sequence']:
                    result, annotation = annotation, next(results, None)
                row = GestureDataset._frame(self, capture, result)
                pair = Pair(images, capture['source_timestamp'], capture['receive_timestamp'],
                            capture['sequence'], capture['metadata'])
                yield pair, capture['capture_time'], _hands(row['observation.gesture_json'], result is not None), row
        finally:
            if handle is not None:
                handle.close()

    def image_hashes(self):
        cameras = [ch['camera'] for ch in self.profile['channels']]
        for row in _rows(self.root / 'raw/capture.jsonl'):
            hashes = row['metadata'].get('deferred_rgb_sha256')
            if not isinstance(hashes, list) or len(hashes) != 2:
                raise ValueError('Missing captured RGB hashes during final validation')
            yield row['sequence'], dict(zip(cameras, hashes))


def open_gesture_reader(root):
    root = Path(root)
    extra = json.loads((root / 'meta/ego2.json').read_text())
    marker = extra.get('deferred_video', {})
    if marker and not marker.get('materialized'):
        return RawGestureReplay(root)
    return DatasetReplay(root)
