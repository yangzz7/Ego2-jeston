"""NumPy/ctypes interface to the portable Cyperstereo stereo ISP.

One :class:`Isp` instance should live for the whole capture session, so its
automatic white-balance history persists. Input is the native interleaved
Bayer transport, *not* ordinary YUYV pixels or an already processed RGB frame.
The library never opens a camera. Output order is C1 then C2, both BGR8.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading

import numpy as np

__all__ = ["Isp", "IspError"]

_ROOT = Path(__file__).resolve().parent.parent
_HEIGHT = 1024
_WIDTH = 1280
_U8_POINTER = ctypes.POINTER(ctypes.c_uint8)
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


class IspError(RuntimeError):
    """A native ISP operation failed; the message includes its C ABI error."""


def _path(value: str | os.PathLike[str]) -> Path:
    result = Path(value).expanduser().resolve()
    if b"\0" in os.fsencode(result):
        raise ValueError("A path must not contain a NUL byte")
    return result


def _bind(library: ctypes.CDLL) -> None:
    library.cisp_api_version.argtypes = []
    library.cisp_api_version.restype = ctypes.c_int
    library.cisp_last_error.argtypes = []
    library.cisp_last_error.restype = ctypes.c_char_p
    library.cisp_create.argtypes = [ctypes.c_char_p]
    library.cisp_create.restype = ctypes.c_void_p
    library.cisp_destroy.argtypes = [ctypes.c_void_p]
    library.cisp_destroy.restype = None
    library.cisp_load_config.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    library.cisp_load_config.restype = ctypes.c_int
    library.cisp_reset.argtypes = [ctypes.c_void_p]
    library.cisp_reset.restype = ctypes.c_int
    library.cisp_process_transport.argtypes = [
        ctypes.c_void_p,
        _U8_POINTER, ctypes.c_size_t, ctypes.c_size_t,
        _U8_POINTER, ctypes.c_size_t, ctypes.c_size_t,
        _U8_POINTER, ctypes.c_size_t, ctypes.c_size_t,
        _DOUBLE_POINTER,
    ]
    library.cisp_process_transport.restype = ctypes.c_int


class Isp:
    """Stateful stereo ISP with serialized native calls.

    ``config=None`` loads ``configs/s200032.yaml`` beside the package's
    ``python`` directory. Library resolution is: explicit ``library``, then
    ``CYPERSTEREO_ISP_LIBRARY``, then ``build/libcyperstereo_isp_c.so``.

    ``process_transport`` accepts uint8 arrays of shape ``(1024,1280,2)`` or
    ``(1024,2560)``. Each pixel's low byte is C1 and high byte is C2. Preserve
    the final metadata row. A real YUV camera frame is not valid input.

    Use a context manager or call ``close()``. A reentrant lock covers all
    native state access, including close, because ctypes releases the GIL
    during calls. The caller must not modify input pixels concurrently with
    processing. Outputs own their memory and remain valid after the next call.
    """

    def __init__(self, config: str | os.PathLike[str] | None = None,
                 library: str | os.PathLike[str] | None = None, *, backend: str | None = None):
        self._lock = threading.RLock()
        self._handle: int | None = None
        self._last_white_balance_rgb: np.ndarray | None = None
        chosen_library = library if library is not None else os.environ.get("CYPERSTEREO_ISP_LIBRARY")
        if chosen_library is None:
            chosen_library = _ROOT / "build" / "libcyperstereo_isp_c.so"
        self.library_path = _path(chosen_library)
        self.config_path = _path(config if config is not None else _ROOT / "configs" / "s200032.yaml")
        try:
            self._library = ctypes.CDLL(str(self.library_path))
        except OSError as error:
            raise IspError(f"Cannot load ISP library {self.library_path}: {error}") from error
        try:
            # Check the version before binding the rest of the ABI. A library
            # with another version may have incompatible function signatures.
            self._library.cisp_api_version.argtypes = []
            self._library.cisp_api_version.restype = ctypes.c_int
            version = self._library.cisp_api_version()
            if version != 1:
                raise IspError(f"Unsupported ISP ABI version {version}; this wrapper requires 1")
            _bind(self._library)
        except AttributeError as error:
            raise IspError(f"Missing ISP C ABI symbol in {self.library_path}: {error}") from error
        if backend is None:
            handle = self._library.cisp_create(os.fsencode(self.config_path))
        else:
            if backend not in ('cpu', 'cuda'):
                raise ValueError('Display backend must be cpu or cuda')
            try:
                create = self._library.cisp_create_display
            except AttributeError as error:
                raise IspError('This library does not provide the display backend extension') from error
            create.argtypes = [ctypes.c_char_p, ctypes.c_int]
            create.restype = ctypes.c_void_p
            handle = create(os.fsencode(self.config_path), int(backend == 'cuda'))
        if not handle:
            self._raise_native("create")
        self._handle = handle
        self.backend = backend or 'cpu'

    def _raise_native(self, operation: str) -> None:
        # The error is thread-local: read it immediately on the calling thread.
        message = self._library.cisp_last_error()
        detail = message.decode("utf-8", errors="replace") if message else "unspecified native error"
        raise IspError(f"ISP {operation} failed: {detail}")

    def _require_open(self) -> int:
        if self._handle is None:
            raise IspError("ISP is closed")
        return self._handle

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    @property
    def last_white_balance_rgb(self) -> np.ndarray | None:
        """Copy of actual gains as ``[C1/C2, R/G/B]``, or None before a frame.

        A backend which cannot expose internal gains may report NaN values;
        those are left intact rather than presented as measured unit gains.
        Loading a configuration or resetting clears this observation.
        """
        with self._lock:
            value = self._last_white_balance_rgb
            return None if value is None else value.copy()

    def process_transport(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return independent full-size ``(c1_bgr, c2_bgr)`` uint8 images.

        Non-contiguous input is copied; contiguous input is borrowed read-only
        for the duration of the native call. No color conversion is applied in
        Python. The native decoder validates hardware/firmware metadata.
        """
        with self._lock:
            handle = self._require_open()
            if not isinstance(frame, np.ndarray):
                raise TypeError("frame must be a NumPy ndarray")
            if frame.dtype != np.uint8:
                raise TypeError(f"frame must have dtype uint8, got {frame.dtype}")
            if frame.shape not in ((_HEIGHT, _WIDTH, 2), (_HEIGHT, _WIDTH * 2)):
                raise ValueError(f"frame must have shape (1024, 1280, 2) or (1024, 2560), got {frame.shape}")
            transport = np.ascontiguousarray(frame)
            c1 = np.empty((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
            c2 = np.empty_like(c1)
            white_balance = np.empty((2, 3), dtype=np.float64)
            status = self._library.cisp_process_transport(
                handle,
                transport.ctypes.data_as(_U8_POINTER), transport.nbytes, transport.strides[0],
                c1.ctypes.data_as(_U8_POINTER), c1.nbytes, c1.strides[0],
                c2.ctypes.data_as(_U8_POINTER), c2.nbytes, c2.strides[0],
                white_balance.ctypes.data_as(_DOUBLE_POINTER),
            )
            if status != 0:
                self._raise_native("process_transport")
            self._last_white_balance_rgb = white_balance
            return c1, c2

    def load_config(self, filename: str | os.PathLike[str]) -> None:
        """Load a validated YAML configuration into this existing instance."""
        with self._lock:
            handle = self._require_open()
            path = _path(filename)
            if self._library.cisp_load_config(handle, os.fsencode(path)) != 0:
                self._raise_native("load_config")
            self.config_path = path
            self._last_white_balance_rgb = None

    def reset(self) -> None:
        """Reset temporal ISP state while retaining the loaded configuration."""
        with self._lock:
            if self._library.cisp_reset(self._require_open()) != 0:
                self._raise_native("reset")
            self._last_white_balance_rgb = None

    def close(self) -> None:
        """Wait for any active native call, then release this instance once."""
        with self._lock:
            handle, self._handle = self._handle, None
            if handle is not None:
                self._library.cisp_destroy(handle)

    def __enter__(self) -> Isp:
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # __init__ can fail before the native library is loaded; explicit
            # close/context management remains the primary lifecycle contract.
            pass
