"""Packaging regressions: shared calibration and entry points outside the project."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ego2.common.paths import ROOT
from ego2.gesture.config import load_config as load_gesture
from ego2.reconstruction.calibration import load_config as load_reconstruction


class StandaloneTests(unittest.TestCase):
    def test_both_apps_use_the_same_local_camera_bundle(self):
        gesture, _ = load_gesture()
        reconstruction = load_reconstruction()
        self.assertEqual(gesture["input"]["serial"], reconstruction["serial"])
        self.assertTrue(Path(gesture["calibration"]["file"]).samefile(reconstruction["calibration"]))
        paths = [gesture["calibration"]["file"], gesture["gesture"]["model"]]
        paths.extend(reconstruction[k] for k in ("calibration", "imu_calibration", "vocabulary", "orb_source"))
        for path in paths:
            self.assertTrue(Path(path).exists(), path)
            self.assertTrue(Path(path).resolve().is_relative_to(ROOT), path)

    def test_launchers_work_outside_project_without_pythonpath(self):
        env = {**os.environ, "EGO2_PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as unrelated:
            for script in ("run_gesture.sh", "run_reconstruction.sh", "run_hand_pose.sh", "run_aprilgrid.sh"):
                result = subprocess.run([str(ROOT/script), "--help"], cwd=unrelated, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--config", result.stdout)
                self.assertIn("--sn", result.stdout)

    def test_reconstruction_viewer_has_no_gesture_dependency(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import ego2.reconstruction.visualizer; "
            "assert not any(k == 'ego2.gesture' or k.startswith('ego2.gesture.') for k in sys.modules)"],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_multi_camera_launcher_works_outside_project_without_pythonpath(self):
        env = {**os.environ, "EGO2_PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as unrelated:
            result = subprocess.run([str(ROOT/"run_multi_vio.sh"), "--help"], cwd=unrelated,
                                    env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--config", "--no-setup", "--record"):
            self.assertIn(option, result.stdout)
        self.assertNotIn("--chest-sn", result.stdout)
        self.assertNotIn("--hand-sn", result.stdout)


if __name__ == "__main__":
    unittest.main()
