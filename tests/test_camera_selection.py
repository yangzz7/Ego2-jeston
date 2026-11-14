"""A selected USB identity must travel with its own calibration in every app."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from ego2.common.camera import load_camera
from ego2.common.config import read_yaml
from ego2.common.paths import ROOT
from ego2.gesture.config import load_config as load_gesture, DEFAULT_CONFIG as GESTURE_CONFIG
from ego2.aprilgrid.config import load_config as load_aprilgrid
from ego2.hand_pose.config import load_config as load_hand_pose
from ego2.reconstruction.calibration import load_config as load_reconstruction
from ego2.reconstruction.app import create_session


LOADERS = (load_gesture, load_aprilgrid, load_hand_pose, load_reconstruction)


def profile(loader, sn=None):
    result = loader(sn=sn)
    return result if isinstance(result, dict) else result[0]


class CameraSelectionTests(unittest.TestCase):
    def test_default_and_explicit_selection_use_complete_camera_bundle(self):
        for loader in LOADERS:
            for requested, serial in ((None, "s200032"), ("s200032", "s200032"),
                                      ("s200002", "s200002"), ("s200100", "s200100")):
                with self.subTest(loader=loader.__module__, sn=requested):
                    cfg = profile(loader, requested)
                    folder = ROOT / "cameras" / serial
                    self.assertEqual(Path(cfg["camera"]), folder / "camera.yaml")
                    if loader is load_reconstruction:
                        self.assertEqual(cfg["serial"], serial)
                        self.assertEqual(Path(cfg["calibration"]), folder / "camchain-imucam.yaml")
                        imu = "imu-calibrated.yaml"
                        self.assertEqual(Path(cfg["imu_calibration"]), folder / imu)
                    else:
                        self.assertEqual(cfg["input"]["serial"], serial)
                        self.assertEqual(cfg["input"]["device"], "auto")
                        self.assertEqual(Path(cfg["calibration"]["file"]), folder / "camchain-imucam.yaml")
                    if loader is load_hand_pose:
                        self.assertEqual(Path(cfg["bundle_calibration"]), folder / "wrist_bundles.json")
                    if loader is load_aprilgrid:
                        isp = cfg["viewer"]["display_isp"]
                        if serial == "s200100":
                            self.assertFalse(isp["enabled"])
                            self.assertIn(serial, isp["disabled_reason"])
                        else:
                            self.assertTrue(isp["enabled"])
                            self.assertEqual(Path(isp["config"]).name, serial + ".yaml")
        for serial in ("s200002", "s200100"):
            _, cameras = load_gesture(sn=serial)
            raw = read_yaml(ROOT / "cameras" / serial / "camchain-imucam.yaml")
            for name in ("cam0", "cam1"):
                self.assertAlmostEqual(cameras[name].K[0, 0], raw[name]["intrinsics"][0])
                np.testing.assert_array_equal(cameras[name].D, raw[name]["distortion_coeffs"])

    def test_explicit_sn_overrides_stale_device_identity_and_geometry(self):
        cfg = read_yaml(GESTURE_CONFIG)
        cfg["camera"] = str(ROOT / "cameras/s200032/camera.yaml")
        cfg["input"] = dict(type="uvc", serial="s200032", device="/dev/video0", width=1, height=1,
                            bayer="wrong", acceleration_scale=1, fps=15)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gesture.yaml"
            path.write_text(yaml.safe_dump(cfg))
            selected, _ = load_gesture(path, sn="s200002")
        hardware = selected["input"]
        self.assertEqual(hardware["type"], "cyper")
        self.assertEqual((hardware["serial"], hardware["device"]), ("s200002", "auto"))
        self.assertEqual((hardware["width"], hardware["height"], hardware["bayer"]), (1280, 1024, "auto"))
        self.assertEqual(hardware["acceleration_scale"], 9.7887)
        self.assertEqual(hardware["fps"], 15)

    def test_new_camera_can_be_registered_without_code_changes(self):
        bundle = load_camera(sn="s200002")
        bundle["serial"] = "s200099"
        with tempfile.TemporaryDirectory() as directory, patch("ego2.common.camera.CAMERAS", Path(directory)):
            folder = Path(directory) / "s200099"
            folder.mkdir()
            (folder / "camera.yaml").write_text(yaml.safe_dump(bundle))
            for loader in LOADERS:
                with self.subTest(loader=loader.__module__):
                    cfg = profile(loader, "s200099")
                    self.assertEqual(Path(cfg["camera"]), folder / "camera.yaml")
                    serial = cfg["serial"] if loader is load_reconstruction else cfg["input"]["serial"]
                    self.assertEqual(serial, "s200099")
                    if loader is load_aprilgrid:
                        isp = cfg["viewer"]["display_isp"]
                        self.assertFalse(isp["enabled"])
                        self.assertIn("s200099", isp["disabled_reason"])
                    if loader is load_hand_pose:
                        self.assertEqual(Path(cfg["bundle_calibration"]), folder / "wrist_bundles.json")

    def test_missing_invalid_and_mislabeled_serials_never_fall_back(self):
        with tempfile.TemporaryDirectory() as directory, patch("ego2.common.camera.CAMERAS", Path(directory)):
            for loader in LOADERS:
                with self.subTest(loader=loader.__module__):
                    with self.assertRaisesRegex(ValueError, "No camera profile.*s200099"):
                        profile(loader, "s200099")
            for serial in ("", "../s200032", "/tmp/camera", "s200002/extra"):
                with self.assertRaisesRegex(ValueError, "Invalid --sn"):
                    load_camera(sn=serial)
            folder = Path(directory) / "s200099"
            folder.mkdir()
            (folder / "camera.yaml").write_text("serial: s200032\n")
            with self.assertRaisesRegex(ValueError, "does not match requested SN"):
                load_camera(sn="s200099")
            (folder / "camera.yaml").write_text("serial: s200099\ncalibration: missing.yaml\n")
            with self.assertRaisesRegex(ValueError, "Missing camera parameter file"):
                load_camera(sn="s200099")

    def test_reconstruction_session_preserves_selected_calibration_and_generates_orb(self):
        cfg = load_reconstruction(sn="s200002")
        raw = read_yaml(cfg["calibration"])
        imu = read_yaml(cfg["imu_calibration"])["imu0"]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(session=str(Path(directory) / "session"), demo=False, stereo_only=False)
            session, effective, shift = create_session(cfg, args)
            self.assertEqual((session / "calibration.yaml").read_bytes(), Path(cfg["calibration"]).read_bytes())
            self.assertEqual((session / "imu_calibration.yaml").read_bytes(), Path(cfg["imu_calibration"]).read_bytes())
            self.assertEqual(effective["serial"], "s200002")
            fs = cv2.FileStorage(str(session / "orb_settings.yaml"), cv2.FILE_STORAGE_READ)
            try:
                self.assertAlmostEqual(fs.getNode("Camera.fx").real(), raw["cam1"]["intrinsics"][0])
                self.assertAlmostEqual(fs.getNode("Camera2.fx").real(), raw["cam0"]["intrinsics"][0])
                np.testing.assert_allclose(fs.getNode("Tbc").mat() @ raw["cam1"]["T_cam_imu"], np.eye(4), atol=1e-7)
                np.testing.assert_allclose(fs.getNode("Tlr").mat(), np.array(raw["cam1"]["T_cn_cnm1"])[:3], atol=1e-7)
                self.assertEqual(fs.getNode("IMU.NoiseGyro").real(), imu["gyroscope_noise_density"])
                self.assertEqual(fs.getNode("Camera.RGB").real(), 0)
                self.assertEqual(shift, raw["cam1"]["timeshift_cam_imu"])
            finally:
                fs.release()


class CameraCliTests(unittest.TestCase):
    def test_missing_wrist_geometry_instructions_keep_selected_sn(self):
        from ego2.hand_pose.__main__ import main
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            missing = str(Path(directory) / "wrists.json")
            self.assertEqual(main(["--sn", "s200002", "--bundle-calibration", missing]), 1)
            self.assertIn("--sn s200002", errors.getvalue())
            self.assertIn(str(ROOT / "run_hand_pose.sh"), errors.getvalue())
            self.assertIn(missing, errors.getvalue())

    def test_all_entrypoints_pass_selected_bundle_to_runtime(self):
        for name, options in (("gesture", ["--no-record", "--no-viz"]),
                              ("aprilgrid", ["--no-record", "--no-viz"]),
                              ("hand_pose", ["scan", "--no-viz"]),
                              ("reconstruction", ["--no-viz"])):
            main = importlib.import_module(f"ego2.{name}.__main__").main
            for sn_args, serial in (([], "s200032"), (["--sn", "s200002"], "s200002")):
                with self.subTest(app=name, serial=serial), patch(f"ego2.{name}.app.run") as run, redirect_stdout(StringIO()):
                    self.assertEqual(main(options + sn_args), 0)
                    cfg = run.call_args.args[0]
                    self.assertEqual(Path(cfg["camera"]).parent.name, serial)
            with self.subTest(app=name, missing=True), patch(f"ego2.{name}.app.run") as run, redirect_stderr(StringIO()):
                self.assertEqual(main(options + ["--sn", "unregistered-camera"]), 1)
                run.assert_not_called()

    def test_conflicting_device_and_recording_overrides_are_rejected(self):
        cases = [(name, ["--sn", "s200002", "--device", "/dev/video0"])
                 for name in ("gesture", "aprilgrid", "hand_pose")]
        cases += [("gesture", ["--sn", "s200002", "--replay", "/tmp/existing-session"]),
                  ("reconstruction", ["export", "--sn", "s200002", "--session", "/tmp/existing-session"])]
        for name, args in cases:
            with self.subTest(app=name, args=args), redirect_stderr(StringIO()):
                main = importlib.import_module(f"ego2.{name}.__main__").main
                with self.assertRaises(SystemExit) as caught:
                    main(args)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
