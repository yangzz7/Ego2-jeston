"""Jetson manual Cyper recording: journal RAW now, encode after stop acknowledgement.

The original Source/algorithm RGB is hashed before journaling. Reconstruction
and decoded lossless video must both match it. No encoder runs on this capture
path, including episode rollover. Other sources and --no-raw keep the existing
writer. This module does not change capture FPS, inference or preview.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import GestureDataset, VIDEO_KEYS, _Moments


class _RawImage:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def write(self, image):
        if image.dtype != np.uint8 or tuple(image.shape) != self.shape:
            raise ValueError('RAW recording image shape/dtype changed')

    def close(self):
        pass


class RawGestureDataset(GestureDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra['deferred_video'] = dict(version=1,
            storage='original Cyper RAW; RGB SHA256 per eye per frame',
            finalization='after capture stop acknowledgement, in the parent finalizer')
        self._update_extra()

    def _start_segment(self, images):
        self._segment_count = 0
        self._videos, self._video_stats = [], [_Moments(), _Moments()]
        for key, image in zip(VIDEO_KEYS, images):
            h, w, _ = image.shape
            if key not in self.features:
                self.features[key] = dict(dtype='video', shape=[h, w, 3],
                    names=['height', 'width', 'channels'], info={
                        'video.height': h, 'video.width': w, 'video.codec': 'h264',
                        'video.pix_fmt': 'gbrp', 'video.is_depth_map': False,
                        'video.fps': self.fps, 'video.channels': 3, 'has_audio': False})
            elif self.features[key]['shape'] != list(image.shape):
                raise ValueError('Video resolution changed during capture')
            self._videos.append(_RawImage(image.shape))

    def _capture_pair(self, pair):
        cfg = self.extra['profile']['input']
        raw = pair.raw_transport
        h, w = int(cfg.get('height', 1024)), int(cfg.get('width', 1280))
        if (cfg.get('type') != 'cyper' or not isinstance(raw, np.ndarray)
                or raw.dtype != np.uint8 or raw.shape != (h, w, 2)):
            raise ValueError('Deferred recording requires original uint8 Cyper RAW for every pair')
        if len(pair.images) != 2 or any(im.shape != (h, w, 3) or im.dtype != np.uint8 for im in pair.images):
            raise ValueError('Deferred recording images disagree with saved Cyper dimensions')
        metadata = dict(pair.metadata or {})
        metadata['deferred_rgb_sha256'] = [
            hashlib.sha256(memoryview(np.ascontiguousarray(im))).hexdigest() for im in pair.images]
        # Do not mutate the shared Pair or metadata used by inference/preview.
        super()._capture_pair(replace(pair, metadata=metadata))

    def close(self, error=None):
        if self._closed:
            return
        self._closed = True
        failure = error or self._failure
        try:
            self._finish_segment()
        except BaseException as exc:
            failure = failure or exc
            raise
        finally:
            for handle in (self._capture, self._results, self._segments, self._raw):
                if handle is not None:
                    handle.close()
            # raw_ready is deliberately not complete: there are no videos or
            # published Parquet tables yet. Recovery validates every journal.
            self._update_extra('failed' if failure else 'raw_ready' if self.capture_count else 'empty', failure)

    finalize = close


def finalized_root(root):
    """Materialize a new dataset for assembly, preserving the original RAW cache."""
    root = Path(root)
    extra = json.loads((root / 'meta/ego2.json').read_text(encoding='utf-8'))
    marker = extra.get('deferred_video')
    if not marker:
        return root
    if marker.get('version') == 1 and marker.get('materialized') and extra.get('recording_status') == 'complete':
        return root
    if marker.get('version') != 1 or extra.get('recording_status') != 'raw_ready':
        raise RuntimeError('Deferred recording did not close successfully; RAW cache retained')
    from .recovery import recover_dataset
    destination = root.with_name(root.name + '-encoded')
    print(f'[record] RAW 已保存 {extra["capture_count"]} 对；正在生成并校验无损视频：{root}', flush=True)
    recover_dataset(root, destination)
    return destination
