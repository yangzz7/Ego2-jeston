"""NumPy-only tests of the C ABI wrapper; all frames are synthetic RAW8.

Run after building from the package root:
    python -B tests/test_python.py
    CYPERSTEREO_ISP_LIBRARY=/path/to/libcyperstereo_isp_c.so python -B tests/test_python.py
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from cyperstereo_isp import Isp, IspError  # noqa: E402

LIBRARY = Path(os.environ.get("CYPERSTEREO_ISP_LIBRARY", ROOT / "build" / "libcyperstereo_isp_c.so"))
CONFIG = ROOT / "configs" / "s200032.yaml"


def set_word(frame: np.ndarray, column: int, value: int) -> None:
    frame[-1, column, 0] = value & 255
    frame[-1, column, 1] = value >> 8


def synthetic_transport() -> np.ndarray:
    """HW2/FW6: C1 physical RGGB, C2 physical BGGR, original metadata row."""
    packed = np.zeros((1024, 1280, 2), dtype=np.uint8)
    for eye in range(2):
        rgb = np.full((1024, 1280, 3), 100, dtype=np.uint8)
        rgb[100:260, 100:260] = (210, 45, 30) if eye == 0 else (30, 45, 210)
        rgb[100:260, 300:460] = (30, 45, 210) if eye == 0 else (210, 45, 30)
        rgb[500:650, 700:900] = (240, 240, 240)
        plane = packed[:, :, eye]
        r_site, b_site = ((0, 0), (1, 1)) if eye == 0 else ((1, 1), (0, 0))
        plane[r_site[0]::2, r_site[1]::2] = rgb[r_site[0]::2, r_site[1]::2, 0]
        plane[0::2, 1::2] = rgb[0::2, 1::2, 1]
        plane[1::2, 0::2] = rgb[1::2, 0::2, 1]
        plane[b_site[0]::2, b_site[1]::2] = rgb[b_site[0]::2, b_site[1]::2, 2]
    packed[-1] = 0
    # Shared metadata words across the two byte lanes. Timestamp =
    # day*43200 + seconds + fraction/10000, so 0,17,1250 is valid.
    for col, value in ((0, 2), (1, 6), (2, 0), (3, 1250), (4, 17),
                       (68, 0), (69, 200), (70, 200),
                       (73, 149 << 8), (74, 149 << 8), (77, 64), (78, 64)):
        set_word(packed, col, value)
    return packed


class IspTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not LIBRARY.is_file():
            raise RuntimeError(f"Build the native library first: {LIBRARY}")
        cls.transport = synthetic_transport()

    def make_isp(self, **kwargs) -> Isp:
        return Isp(library=LIBRARY, **kwargs)

    def test_color_order_neutrality_and_input_immutability(self):
        frame = self.transport.copy()
        original = frame.copy()
        frame.flags.writeable = False
        with self.make_isp() as isp:
            self.assertIsNone(isp.last_white_balance_rgb)
            c1, c2 = isp.process_transport(frame)
            for image in (c1, c2):
                self.assertEqual(image.shape, (1024, 1280, 3))
                self.assertEqual(image.dtype, np.uint8)
                self.assertTrue(image.flags.c_contiguous)
                gray = image[400:460, 500:560].astype(np.int16)
                white = image[540:600, 740:800].astype(np.int16)
                self.assertLessEqual(int(np.ptp(gray, axis=2).max()), 2)
                self.assertLessEqual(int(np.ptp(white, axis=2).max()), 2)
                self.assertGreater(float(white.mean()), 210)
            first = c1[180, 180].astype(float)
            second = c2[180, 180].astype(float)
            self.assertGreater(first[2], first[0] * 2, "C1 should contain a red patch in BGR output")
            self.assertGreater(second[0], second[2] * 2, "C2 should contain a blue patch in BGR output")
            gains = isp.last_white_balance_rgb
            self.assertEqual(gains.shape, (2, 3))
            self.assertTrue(np.isfinite(gains).all())
            np.testing.assert_allclose(gains, np.ones((2, 3)), atol=.01)
            gains[:] = 3
            self.assertFalse(np.array_equal(gains, isp.last_white_balance_rgb), "WB property should return a copy")
            retained = c1.copy()
            isp.process_transport(frame)
            np.testing.assert_array_equal(c1, retained)
        np.testing.assert_array_equal(frame, original)

    def test_two_dimensional_and_noncontiguous_input(self):
        padded = np.empty((1024, 1288, 2), np.uint8)
        roi = padded[:, 4:1284]
        roi[:] = self.transport
        self.assertFalse(roi.flags.c_contiguous)
        with self.make_isp() as isp:
            reference = isp.process_transport(self.transport)
            isp.reset()
            two_dimensional = isp.process_transport(self.transport.reshape(1024, 2560))
            isp.reset()
            noncontiguous = isp.process_transport(roi)
        for i in range(2):
            np.testing.assert_array_equal(reference[i], two_dimensional[i])
            np.testing.assert_array_equal(reference[i], noncontiguous[i])

    def test_rejects_python_inputs_and_native_metadata(self):
        with self.make_isp() as isp:
            for bad in (None, [[1, 2]]):
                with self.assertRaises(TypeError):
                    isp.process_transport(bad)
            for dtype in (np.uint16, np.float32, np.int8):
                with self.assertRaises(TypeError):
                    isp.process_transport(self.transport.astype(dtype))
            for shape in ((1024, 1280), (512, 2560, 2), (1024, 1280, 3), (1024 * 2560,)):
                with self.assertRaises(ValueError):
                    isp.process_transport(np.zeros(shape, np.uint8))
            for column, value in ((0, 1), (1, 8), (3, 10000)):
                bad = self.transport.copy()
                set_word(bad, column, value)
                with self.assertRaises(IspError) as failure:
                    isp.process_transport(bad)
                self.assertIn("process_transport", str(failure.exception))
            # A failed call must not prevent a subsequent valid frame.
            self.assertEqual(isp.process_transport(self.transport)[0].shape, (1024, 1280, 3))

    def test_config_reset_close_and_failure_recovery(self):
        isp = self.make_isp()
        isp.process_transport(self.transport)
        isp.reset()
        self.assertIsNone(isp.last_white_balance_rgb)
        isp.process_transport(self.transport)
        isp.load_config(CONFIG)
        self.assertIsNone(isp.last_white_balance_rgb)
        with tempfile.TemporaryDirectory(prefix="cyper-isp-python-") as directory:
            minimal = Path(directory) / "minimal.yaml"
            source_config = CONFIG.read_text(encoding="utf-8")
            self.assertIn("mode: tunable", source_config)
            minimal.write_text(source_config.replace("mode: tunable", "mode: minimal", 1), encoding="utf-8")
            isp.load_config(minimal)
            uncorrected = isp.process_transport(self.transport)
            np.testing.assert_array_equal(uncorrected[0][430, 530], [100, 100, 100])
            isp.reset()
            for actual, expected in zip(isp.process_transport(self.transport), uncorrected):
                np.testing.assert_array_equal(actual, expected)
            isp.load_config(CONFIG)
            corrected = isp.process_transport(self.transport)
            self.assertGreater(float(corrected[0][430, 530].mean()), 120, "loaded sRGB configuration had no effect")
            missing = Path(directory) / "missing.yaml"
            with self.assertRaises(IspError):
                isp.load_config(missing)
            self.assertEqual(isp.config_path, CONFIG.resolve())
            self.assertEqual(isp.process_transport(self.transport)[1].shape, (1024, 1280, 3))
            bad = Path(directory) / "invalid.yaml"
            bad.write_text("%YAML:1.0\n---\nschema_version: 999\n", encoding="utf-8")
            with self.assertRaises(IspError):
                self.make_isp(config=bad)
        isp.close()
        isp.close()
        self.assertTrue(isp.closed)
        for operation in (lambda: isp.process_transport(self.transport), isp.reset,
                          lambda: isp.load_config(CONFIG), isp.__enter__):
            with self.assertRaisesRegex(IspError, "closed"):
                operation()
        with self.make_isp() as another:
            self.assertFalse(another.closed)
        self.assertTrue(another.closed)

    def test_library_selection_and_abi_version(self):
        with mock.patch.dict(os.environ, {"CYPERSTEREO_ISP_LIBRARY": str(LIBRARY.resolve())}):
            with Isp() as isp:
                self.assertEqual(isp.library_path, LIBRARY.resolve())
        with mock.patch.dict(os.environ, {"CYPERSTEREO_ISP_LIBRARY": "/missing/ignored.so"}):
            with self.make_isp() as isp:
                self.assertEqual(isp.library_path, LIBRARY.resolve())
        with self.assertRaisesRegex(IspError, "Cannot load"):
            Isp(library="/missing/isp-library.so")
        incompatible = mock.Mock()
        incompatible.cisp_api_version.return_value = 99
        with mock.patch.object(ctypes, "CDLL", return_value=incompatible):
            with self.assertRaisesRegex(IspError, "ABI version 99"):
                Isp(library=LIBRARY)
        incompatible.cisp_create.assert_not_called()

    def test_close_waits_for_inflight_process(self):
        isp = self.make_isp()
        native = isp._library.cisp_process_transport
        entered, release, close_finished = threading.Event(), threading.Event(), threading.Event()
        errors = []

        def delayed_native(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release processing call")
            return native(*args)

        def process():
            try:
                isp.process_transport(self.transport)
            except BaseException as error:
                errors.append(error)

        def close():
            try:
                isp.close()
                close_finished.set()
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(isp._library, "cisp_process_transport", side_effect=delayed_native):
            processing = threading.Thread(target=process, daemon=True)
            closing = threading.Thread(target=close, daemon=True)
            try:
                processing.start()
                self.assertTrue(entered.wait(5))
                closing.start()
                self.assertFalse(close_finished.wait(.05), "close overtook an active native call")
            finally:
                release.set()
                processing.join(10)
                if closing.ident is not None:
                    closing.join(10)
            self.assertFalse(processing.is_alive())
            self.assertFalse(closing.is_alive())
        self.assertFalse(errors, repr(errors))
        self.assertTrue(close_finished.is_set())
        self.assertTrue(isp.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
