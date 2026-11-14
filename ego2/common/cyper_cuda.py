"""Exact CUDA conversion for live S2 transport; metadata stays on the CPU."""
import ctypes as C
import os
from pathlib import Path
import threading
import numpy as np
from .cyper_metadata import metadata_words

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "build/cyper_capture/libego2_cyper_cuda.so"
PHASES = {"RG": 0, "BG": 1, "GR": 2, "GB": 3}


def capture_backend():
    # Keep non-Jetson/offline environments usable without a CUDA installation.
    backend = os.environ.get("EGO2_CAPTURE_BACKEND", "cuda" if Path("/etc/nv_tegra_release").exists() else "cpu").lower()
    if backend not in ("cuda", "cpu"):
        raise ValueError("EGO2_CAPTURE_BACKEND must be cuda or cpu")
    return backend


class CudaDecoder:
    def __init__(self, width, height):
        self.width, self.height = int(width), int(height)
        self.lock = threading.Lock()
        self.handle = None
        path = os.environ.get("EGO2_CAPTURE_CUDA_LIBRARY", str(LIBRARY))
        try:
            self.lib = C.CDLL(path)
        except OSError as exc:
            raise RuntimeError("CUDA capture library unavailable; run python -m ego2.common.build_capture_cuda. "
                               "Explicit rollback: EGO2_CAPTURE_BACKEND=cpu. " + str(exc)) from exc
        self.lib.ego2_cyper_create.argtypes = [C.c_int, C.c_int, C.c_void_p, C.c_size_t]
        self.lib.ego2_cyper_create.restype = C.c_void_p
        self.lib.ego2_cyper_destroy.argtypes = [C.c_void_p]
        self.lib.ego2_cyper_destroy.restype = None
        self.lib.ego2_cyper_decode.argtypes = [C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_int,
                                               C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
        self.lib.ego2_cyper_decode.restype = C.c_int
        self.lib.ego2_cyper_decode_balanced.argtypes = self.lib.ego2_cyper_decode.argtypes
        self.lib.ego2_cyper_decode_balanced.restype = C.c_int
        error = C.create_string_buffer(1024)
        self.handle = self.lib.ego2_cyper_create(self.width, self.height, error, len(error))
        if not self.handle:
            raise RuntimeError("CUDA capture initialization failed: " + error.value.decode())

    def decode(self, frame, bayer="auto", *, white_balance=False):
        words = metadata_words(frame, self.width, self.height)
        hw, sw = int(words[0]), int(words[1])
        if hw != 2 or sw not in (3,4,5,6,7):
            raise ValueError(f"Unsupported/invalid SmartSens metadata {hw}/{sw}")
        phases = (["BG" if sw >= 4 else "RG", "RG"] if bayer == "auto" else bayer)
        if len(phases) != 2 or any(p not in PHASES for p in phases):
            raise ValueError("bayer must be auto or two OpenCV code names: RG/BG/GR/GB")
        raw = np.ascontiguousarray(frame).reshape(self.height, self.width, 2)
        # Own each returned image: capture, inference and recording can retain it.
        a = np.empty((self.height, self.width, 3), np.uint8)
        b = np.empty_like(a)
        error = C.create_string_buffer(1024)
        with self.lock:
            if not self.handle:
                raise RuntimeError("CUDA capture decoder is closed")
            decode = self.lib.ego2_cyper_decode_balanced if white_balance else self.lib.ego2_cyper_decode
            status = decode(self.handle, raw.ctypes.data, raw.strides[0],
                PHASES[phases[0]], PHASES[phases[1]], a.ctypes.data, a.strides[0],
                b.ctypes.data, b.strides[0], error, len(error))
        if status:
            raise RuntimeError("CUDA capture conversion failed: " + error.value.decode())
        return (a,b), (hw,sw)

    def close(self):
        with self.lock:
            if self.handle:
                self.lib.ego2_cyper_destroy(self.handle)
                self.handle = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
