"""Offline readiness checks must never create a dataset or open hardware."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import cv2
import yaml

from ego2.gesture.config import load_config
from ego2.gesture.preflight import _dependencies, _ffmpeg, format_report, main, preflight


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.target = self.base / "future/session_002"
        self.model = self.base / "model.task"
        with zipfile.ZipFile(self.model, "w") as archive:
            archive.writestr("hand_landmarker.task", b"readable landmark fixture")
            archive.writestr("hand_gesture_recognizer.task", b"readable gesture fixture")
        block = dict(camera_model="pinhole", distortion_model="equidistant", intrinsics=[20, 20, 16, 12],
                     distortion_coeffs=[0, 0, 0, 0], resolution=[32, 24])
        calibration = dict(cam0=block, cam1={**block, "T_cn_cnm1": [[1, 0, 0, 0.1], [0, 1, 0, 0],
                                                                 [0, 0, 1, 0], [0, 0, 0, 1]]})
        self.calibration = self.base / "calibration.yaml"
        self.calibration.write_text(yaml.safe_dump(calibration))
        self.config = self.base / "profile.yaml"
        self.config.write_text(yaml.safe_dump(dict(
            calibration=dict(file="calibration.yaml", format="kalibr"),
            input=dict(type="cyper", device="/dev/ego2-test-device-absent", width=32, height=24, fps=30),
            gesture=dict(model="model.task"), channels=[dict(camera="cam0"), dict(camera="cam1")],
            inference_view=dict(width=32, height=32, hfov=75, yaw=0, pitch=0))))
        self.profile, _ = load_config(self.config)

    def run_checks(self, **kwargs):
        with patch("ego2.gesture.preflight._dependencies", return_value=dict(versions={"test": "1"})):
            with patch("ego2.gesture.preflight._ffmpeg", return_value=dict(lossless_roundtrip=True)):
                return preflight(self.profile, self.target, **kwargs)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs real FFmpeg")
    def test_real_encoder_probe_and_complete_check_never_open_hardware_or_write_target(self):
        before = {path.relative_to(self.base): path.read_bytes() for path in self.base.rglob("*") if path.is_file()}
        with patch("ego2.gesture.preflight._dependencies", return_value=dict(versions={"test": "1"})), \
                patch("cv2.VideoCapture", side_effect=AssertionError("must not open capture")), \
                patch("ego2.common.sources.Source.open", side_effect=AssertionError("must not open source")), \
                patch("ego2.gesture.recognizer.Recognizer.__init__", side_effect=AssertionError("must not initialize inference")):
            report = preflight(self.profile, self.target, min_free_bytes=0)
        self.assertTrue(report["ok"], report)
        self.assertTrue(any("camera_node" in warning for warning in report["warnings"]))
        self.assertFalse(self.target.parent.exists())
        self.assertEqual(before, {path.relative_to(self.base): path.read_bytes()
                                  for path in self.base.rglob("*") if path.is_file()})
        encoder = next(check["details"] for check in report["checks"] if check["name"] == "ffmpeg")
        self.assertTrue(encoder["lossless_roundtrip"])
        self.assertEqual(encoder["probe_frames"], 2)
        self.assertIn("Camera and model inference were not started", format_report(report))
        json.dumps(report)

    def test_existing_destination_and_dangling_symlink_fail_without_changes(self):
        self.target.parent.mkdir()
        for kind in ("directory", "dangling_symlink"):
            with self.subTest(kind=kind):
                if kind == "directory":
                    self.target.mkdir()
                else:
                    self.target.symlink_to(self.base / "absent")
                report = self.run_checks()
                self.assertFalse(report["ok"])
                self.assertTrue(any("already exists" in message for message in report["errors"]))
                if kind == "directory":
                    self.target.rmdir()
                else:
                    self.target.unlink()

    def test_unwritable_parent_is_reported_without_test_file_creation(self):
        with patch("ego2.gesture.preflight.os.access", return_value=False):
            report = self.run_checks(check_camera=False)
        self.assertFalse(report["ok"])
        self.assertTrue(any("not writable/searchable" in message for message in report["errors"]))
        self.assertFalse(self.target.parent.exists())

    def test_required_camera_visibility_is_error_before_creating_dataset(self):
        report = self.run_checks(require_camera=True)
        self.assertFalse(report["ok"])
        self.assertTrue(any("camera_node: Camera node is not visible in this environment" in message
                            for message in report["errors"]))
        self.assertEqual(report["warnings"], [])
        self.assertFalse(self.target.parent.exists())

    def test_low_disk_space_and_requested_duration_are_checked(self):
        usage = SimpleNamespace(total=10 * 1024**3, used=0, free=512 * 1024**2)
        with patch("ego2.gesture.preflight.shutil.disk_usage", return_value=usage):
            report = self.run_checks(check_camera=False)
        self.assertFalse(report["ok"])
        self.assertTrue(any("Insufficient disk space" in message for message in report["errors"]))
        with patch("ego2.gesture.preflight.shutil.disk_usage", return_value=usage):
            report = self.run_checks(check_camera=False, min_free_bytes=0, seconds=1)
        self.assertTrue(report["ok"], report)
        storage = next(check["details"] for check in report["checks"] if check["name"] == "storage")
        self.assertEqual(storage["raw_bytes_per_second"], 32 * 24 * 2 * 30)
        self.assertGreater(storage["estimated_recording_bytes"], storage["raw_bytes_per_second"])

    def test_missing_model_and_corrupt_calibration_are_required_failures(self):
        self.model.unlink()
        self.calibration.write_text("invalid: calibration\n")
        report = self.run_checks(check_camera=False)
        self.assertFalse(report["ok"])
        self.assertTrue(any("model:" in message for message in report["errors"]))
        self.assertTrue(any("configuration:" in message for message in report["errors"]))

    def test_corrupt_model_bundle_is_rejected_without_initializing_graph(self):
        self.model.write_bytes(b"broken model")
        report = self.run_checks(check_camera=False)
        self.assertFalse(report["ok"])
        self.assertTrue(any("Unreadable gesture model task bundle" in message for message in report["errors"]))

    def test_missing_dependency_is_reported_and_no_viz_skips_rerun_import(self):
        imported = []
        def importing(name):
            imported.append(name)
            if name == "pyarrow":
                raise ImportError("test missing Arrow")
            return SimpleNamespace(__version__="test")
        with patch("ego2.gesture.preflight.importlib.import_module", side_effect=importing):
            with self.assertRaisesRegex(RuntimeError, "pyarrow.*test missing Arrow"):
                _dependencies(False)
        self.assertEqual(imported[0], "cv2")
        self.assertNotIn("rerun", imported)
        with patch("ego2.gesture.preflight._dependencies", side_effect=RuntimeError("pyarrow unavailable")), \
                patch("ego2.gesture.preflight._ffmpeg", return_value={}):
            report = preflight(self.profile, self.target, check_camera=False)
        self.assertFalse(report["ok"])
        self.assertTrue(any("dependencies: pyarrow unavailable" in message for message in report["errors"]))

    def test_missing_and_failing_ffmpeg_return_clear_errors(self):
        with patch("ego2.gesture.preflight.shutil.which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "ffmpeg is missing"):
                _ffmpeg()
        with patch("ego2.gesture.preflight.shutil.which", return_value="/test/ffmpeg"), \
                patch("ego2.gesture.preflight.subprocess.run", return_value=SimpleNamespace(
                    returncode=1, stderr=b"Unknown encoder libx264rgb")):
            with self.assertRaisesRegex(RuntimeError, "libx264rgb encode failed.*Unknown encoder"):
                _ffmpeg()

    def test_standalone_cli_returns_nonzero_on_failed_check(self):
        with patch("ego2.gesture.preflight._dependencies", return_value={}), \
                patch("ego2.gesture.preflight._ffmpeg", side_effect=RuntimeError("encoder unavailable")), \
                redirect_stdout(io.StringIO()) as output:
            status = main(["--config", str(self.config), "--dataset", str(self.target), "--no-viz", "--json"])
        self.assertEqual(status, 1)
        report = json.loads(output.getvalue())
        self.assertFalse(report["ok"])
        self.assertTrue(any("encoder unavailable" in error for error in report["errors"]))
        self.assertFalse(self.target.parent.exists())


if __name__ == "__main__":
    unittest.main()
