"""Marker-file relocation must preserve camera paths and legacy profiles."""
from pathlib import Path
import tempfile
import unittest

import yaml

from ego2.common.config import read_yaml
from ego2.hand_pose.config import DEFAULT_CONFIG, load_config


class MarkerConfigTests(unittest.TestCase):
    def profile_with_absolute_camera_paths(self):
        profile = read_yaml(DEFAULT_CONFIG)
        profile.setdefault("bundle_calibration", str(Path(profile["camera"]).parent / "wrist_bundles.json"))
        for key in ("camera", "bundle_calibration"):
            profile[key] = str((DEFAULT_CONFIG.parent / profile[key]).resolve())
        return profile

    def test_external_parameters_match_legacy_inline_profile(self):
        expected, _ = load_config()
        expected["camera"] = str((DEFAULT_CONFIG.parent / expected["camera"]).resolve())
        legacy = self.profile_with_absolute_camera_paths()
        markers = read_yaml(DEFAULT_CONFIG.parent / legacy.pop("marker_config"))
        legacy.update(markers)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.yaml"
            path.write_text(yaml.safe_dump(legacy), encoding="utf-8")
            actual, _ = load_config(path)
        for key in ("_path", "_base", "marker_config"):
            expected.pop(key, None)
            actual.pop(key, None)
        self.assertEqual(actual, expected)

    def test_relative_reference_uses_profile_directory_and_keeps_inline_overrides(self):
        profile = self.profile_with_absolute_camera_paths()
        markers = read_yaml(DEFAULT_CONFIG.parent / profile["marker_config"])
        markers["targets"][0]["size_m"] = 0.031
        profile["marker_config"] = "../targets/custom.yaml"
        profile["border_bits"] = 1
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "profiles").mkdir()
            (root / "targets").mkdir()
            marker_path = root / "targets/custom.yaml"
            marker_path.write_text(yaml.safe_dump(markers), encoding="utf-8")
            path = root / "profiles/hand.yaml"
            path.write_text(yaml.safe_dump(profile), encoding="utf-8")
            cfg, _ = load_config(path)
        self.assertEqual(cfg["marker_config"], str(marker_path))
        self.assertEqual(cfg["targets"][0]["size_m"], 0.031)
        self.assertEqual(cfg["border_bits"], 1)
        self.assertEqual(cfg["bundle_calibration"], profile["bundle_calibration"])


if __name__ == "__main__":
    unittest.main()
