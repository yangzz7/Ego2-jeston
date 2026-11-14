"""Bounded, local-only image handoff for disposable gesture previews.

No recording queue uses this transport. The producer publishes complete RAW /
RGB sets under a lock; a consumer either copies the matching generation or
rejects the entire frame, including its annotations. Buffers never alias the
camera, inference, archive or an asynchronously reused shared-memory slot.
"""
from dataclasses import dataclass
from multiprocessing import shared_memory

import cv2
import numpy as np
import rerun as rr


@dataclass
class PreviewImage:
    pixels: object
    quality: int
    rerun_compress: bool = False


def encode_image(image, quality):
    if quality:
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError('Cannot encode Rerun preview image')
        return rr.EncodedImage(contents=encoded, media_type='image/jpeg')
    return rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class PreviewRing:
    SLOTS = 3
    # Two full-resolution BGR images plus the original packed RAW8 stereo pair.
    SLOT_BYTES = 1024 * 1280 * 8

    def __init__(self, spec, *, owner=False):
        self.spec = spec
        self.owner = owner
        self.memory = shared_memory.SharedMemory(name=spec['name'])
        self.lock, self.generations = spec['lock'], spec['generations']
        self.sequence = 0

    @classmethod
    def create(cls, context):
        memory = shared_memory.SharedMemory(create=True, size=cls.SLOTS * cls.SLOT_BYTES)
        try:
            spec = dict(name=memory.name, lock=context.Lock(),
                        generations=context.Array('Q', cls.SLOTS, lock=False))
            return cls(spec, owner=True)
        except BaseException:
            memory.unlink()
            raise
        finally:
            memory.close()

    def write(self, arrays):
        values = [np.asarray(a) for a in arrays]
        if any(a.dtype != np.uint8 for a in values):
            raise ValueError('Preview buffers must be uint8')
        if sum(a.nbytes for a in values) > self.SLOT_BYTES:
            raise ValueError('Preview images exceed the bounded shared-memory slot')
        if not self.lock.acquire(False):
            return None
        try:
            self.sequence += 1
            slot = (self.sequence - 1) % self.SLOTS
            offset = slot * self.SLOT_BYTES
            descriptions = []
            # Invalidate before copying. Publish the generation only once all
            # buffers are complete; even an interrupted write cannot look valid.
            self.generations[slot] = 0
            for array in values:
                destination = np.ndarray(array.shape, np.uint8, buffer=self.memory.buf, offset=offset)
                np.copyto(destination, array)
                descriptions.append((offset - slot * self.SLOT_BYTES, array.shape))
                offset += array.nbytes
            del destination
            self.generations[slot] = self.sequence
            return dict(slot=slot, generation=self.sequence, arrays=descriptions)
        finally:
            self.lock.release()

    def read(self, reference):
        if not self.lock.acquire(False):
            return None
        try:
            slot = reference['slot']
            if slot not in range(self.SLOTS) or self.generations[slot] != reference['generation']:
                return None
            result = []
            for offset, shape in reference['arrays']:
                length = int(np.prod(shape))
                if offset < 0 or length < 0 or offset + length > self.SLOT_BYTES:
                    raise ValueError('Invalid preview buffer descriptor')
                view = np.ndarray(shape, np.uint8, buffer=self.memory.buf,
                                  offset=slot*self.SLOT_BYTES + offset)
                result.append(view.copy())
            return result
        finally:
            self.lock.release()

    def close(self):
        if self.memory is None:
            return
        memory, self.memory = self.memory, None
        memory.close()
        if self.owner:
            try:
                memory.unlink()
            except FileNotFoundError:
                pass


def pack_images(batch, ring):
    """Leave only scalar buffer descriptors in the multiprocessing queue."""
    arrays, logs = [], []
    for path, archetypes, kwargs, times in batch['logs']:
        items = []
        for value in archetypes:
            if isinstance(value, PreviewImage):
                items.append(PreviewImage(len(arrays), value.quality, value.rerun_compress))
                arrays.append(value.pixels)
            else:
                items.append(value)
        logs.append((path, tuple(items), kwargs, times))
    raw_index = None
    if 'display_raw' in batch:
        raw_index = len(arrays)
        arrays.append(batch['display_raw'])
    if not arrays:
        return batch
    reference = ring.write(arrays)
    if reference is None:
        return None
    packed = {k: v for k, v in batch.items() if k != 'display_raw'}
    packed.update(logs=logs, preview_buffers=reference, raw_index=raw_index)
    return packed


def unpack_images(batch, ring):
    if 'preview_buffers' not in batch:
        return batch
    arrays = ring.read(batch['preview_buffers'])
    if arrays is None:
        return None
    resolved = {k: v for k, v in batch.items() if k not in ('preview_buffers', 'raw_index')}
    resolved['logs'] = [
        (path, tuple(PreviewImage(arrays[value.pixels], value.quality, value.rerun_compress)
                     if isinstance(value, PreviewImage) else value for value in archetypes), kwargs, times)
        for path, archetypes, kwargs, times in batch['logs']]
    if batch['raw_index'] is not None:
        resolved['display_raw'] = arrays[batch['raw_index']]
    resolved['owned_preview_buffers'] = True
    return resolved
