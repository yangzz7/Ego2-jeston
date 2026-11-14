"""Read the optional native RAW preview ring without delaying its writer."""
import fcntl
import mmap
from pathlib import Path
import struct

import numpy as np


class DisplayRawReader:
    slot_bytes = 8+1024*1280*2
    slots = 3

    def __init__(self, path):
        self.file = Path(path).open('rb')
        try:
            if Path(path).stat().st_size != self.slots*self.slot_bytes:
                raise ValueError('Invalid display RAW ring size')
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self.file.close()
            raise

    def read(self, sequence):
        if type(sequence) is not int or sequence < 0:
            raise ValueError('Invalid display RAW sequence')
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            offset = (sequence % self.slots)*self.slot_bytes
            if struct.unpack_from('<Q', self.mapping, offset)[0] != sequence+1:
                return None
            return np.ndarray((1024,1280,2), dtype=np.uint8,
                              buffer=self.mapping, offset=offset+8).copy()
        finally:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)

    def close(self):
        self.mapping.close()
        self.file.close()
