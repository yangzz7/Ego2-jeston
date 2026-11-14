"""Synthetic SmartSens transport checks against bundled firmware/SDK semantics."""

from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from ego2.common.cyper_metadata import CyperMetadataDecoder, metadata_words
from ego2.common.config import load_profile
from ego2.common.sources import Pair, Source, decode_cyper


def metadata(software=5, period=0, seconds=12, ticks=1000, width=160):
    words = np.zeros(width, dtype=np.uint16)
    words[:5] = [2, software, period, ticks, seconds]
    return words


def imu(words, index, seconds=12, ticks=1000, motion=(1000, -2000, 3000, -4000, 5000, -6000), temperature=16):
    values = (ticks, seconds, *motion, temperature)
    words[5 + index * 9:14 + index * 9] = [value & 0xffff for value in values]


def transport(words, height=8):
    frame = np.full((height, len(words), 2), 127, dtype=np.uint8)
    frame[-1, :, 0] = words & 255
    frame[-1, :, 1] = words >> 8
    return frame


class MetadataTests(unittest.TestCase):
    def test_configured_acceleration_scale_and_recorded_units(self):
        words = metadata()
        imu(words, 0)
        decoded = CyperMetadataDecoder(acceleration_scale=9.7887).decode_words(words)
        self.assertEqual(decoded["acceleration_scale_m_s2_per_g"], 9.7887)
        sample = decoded["imu_samples"][0]
        np.testing.assert_allclose(sample["acceleration_m_s2"], np.array(sample["acceleration_g"]) * 9.7887)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                CyperMetadataDecoder(acceleration_scale=value)

    def test_little_endian_complete_row_and_original_transport_preserved(self):
        words = metadata()
        words[-1] = 0xabcd  # Outside the interpreted region; must survive.
        frame = transport(words)
        before = frame.copy()
        decoded = CyperMetadataDecoder().decode(frame, len(words), frame.shape[0])
        self.assertEqual(decoded["raw_metadata_words"], words.tolist())
        images, firmware = decode_cyper(frame, len(words), frame.shape[0])
        self.assertEqual(firmware, (2, 5))
        self.assertEqual(images[0].shape, (8, 160, 3))
        np.testing.assert_array_equal(frame, before)

    def test_signed_axes_temperature_scales_and_zero_filled_v5(self):
        words = metadata(period=2)
        imu(words, 0, ticks=950)
        imu(words, 1, ticks=1000, temperature=2047)  # Signed 11-bit -1 -> 22.875 C.
        decoded = CyperMetadataDecoder().decode_words(words)
        self.assertEqual(decoded["image_timestamp_seconds"], 86412.1)
        self.assertEqual(decoded["imu_slot_count"], 13)
        self.assertEqual(len(decoded["imu_slots"]), 13)
        self.assertEqual(len(decoded["imu_samples"]), 2)
        first, second = decoded["imu_samples"]
        self.assertEqual(first["acceleration_raw"], [1000, -2000, 3000])
        self.assertEqual(first["angular_velocity_raw"], [-4000, 5000, -6000])
        np.testing.assert_allclose(first["acceleration_g"], [0.18310546875, -0.3662109375, 0.54931640625])
        np.testing.assert_allclose(first["acceleration_m_s2"], np.array(first["acceleration_g"]) * 9.80665)
        np.testing.assert_allclose(first["angular_velocity_rad_s"], np.array([-4000, 5000, -6000]) * 0.0010652644178602)
        self.assertEqual(first["temperature_c"], 25)
        self.assertEqual(second["temperature_c"], 22.875)
        self.assertEqual(decoded["imu_slots"][2]["rejection_reason"], "zero_filled_slot")
        self.assertFalse(decoded["imu_slots"][-1]["accepted"])

    def test_image_period_rollover_fixes_both_directions(self):
        before = metadata(period=0, seconds=43199, ticks=9990)
        imu(before, 0, seconds=0, ticks=40)
        decoded = CyperMetadataDecoder().decode_words(before)
        self.assertAlmostEqual(decoded["imu_samples"][0]["timestamp_seconds"], 43200.004)
        after = metadata(period=1, seconds=0, ticks=40)
        imu(after, 0, seconds=43199, ticks=9990)
        decoded = CyperMetadataDecoder().decode_words(after)
        self.assertAlmostEqual(decoded["imu_samples"][0]["timestamp_seconds"], 43199.999)

    def test_stale_slots_stop_prefix_but_raw_and_decoded_slots_are_retained(self):
        words = metadata(software=4)
        imu(words, 0, ticks=900)
        imu(words, 1, ticks=950)
        imu(words, 2, ticks=950)
        imu(words, 3, ticks=1000)
        decoded = CyperMetadataDecoder().decode_words(words)
        self.assertEqual(len(decoded["imu_samples"]), 2)
        self.assertEqual(decoded["imu_slots"][2]["rejection_reason"], "timestamp_discontinuity")
        self.assertEqual(decoded["imu_slots"][3]["rejection_reason"], "after_rejected_slot")
        self.assertEqual(decoded["imu_slots"][3]["timestamp_seconds"], 12.1)
        self.assertEqual(decoded["imu_slots"][3]["raw_words"], words[32:41].tolist())

    def test_temperature_continuity_is_retained_between_frames(self):
        decoder = CyperMetadataDecoder()
        words = metadata()
        imu(words, 0, temperature=16)
        self.assertEqual(len(decoder.decode_words(words)["imu_samples"]), 1)
        imu(words, 0, temperature=96)  # +10 C is implausible in one frame.
        decoded = decoder.decode_words(words)
        self.assertEqual(decoded["imu_samples"], [])
        self.assertEqual(decoded["imu_slots"][0]["rejection_reason"], "temperature_discontinuity")
        self.assertEqual(decoded["imu_slots"][0]["temperature_c"], 35)

    def test_temperature_and_image_proximity_filter(self):
        words = metadata()
        imu(words, 0, seconds=11)
        self.assertEqual(CyperMetadataDecoder().decode_words(words)["imu_slots"][0]["rejection_reason"],
                         "timestamp_far_from_image")
        imu(words, 0, temperature=1023)
        self.assertEqual(CyperMetadataDecoder().decode_words(words)["imu_slots"][0]["rejection_reason"],
                         "temperature_out_of_range")

    def test_versioned_ae_layouts_and_python_sensor_order(self):
        for software, marker, line_time, end in [(4, 68, 23.868131868e-6, 81),
                                                 (5, 122, 23.572016461e-6, 135),
                                                 (6, 68, 452 / 15187500, 81),
                                                 (7, 68, 452 / 15187500, 106)]:
            with self.subTest(software=software):
                words = metadata(software=software)
                words[marker] = 0x1234  # Marker value is preserved, not invented/validated.
                words[marker + 1:marker + 5] = [100, 200, 0, 0]
                words[marker + 5:marker + 9] = [0x9604, 0x9702, 0, 0]
                words[marker + 9:marker + 13] = [128, 192, 0, 0]
                decoded = CyperMetadataDecoder().decode_words(words)
                self.assertEqual(decoded["metadata_end_column_exclusive"], end)
                self.assertEqual(decoded["telemetry_marker_raw"], 0x1234)
                self.assertFalse(decoded["telemetry_marker_validated"])
                self.assertEqual([sample["sensor"] for sample in decoded["telemetry"]], ["C1", "C2"])
                first, second = decoded["telemetry"]
                self.assertEqual(first["image_index"], 0)
                self.assertEqual(second["image_index"], 1)
                self.assertAlmostEqual(first["temperature_c"], 27.85)
                self.assertAlmostEqual(second["temperature_c"], 29.35)
                self.assertEqual(first["gain"], 2)
                self.assertEqual(second["gain"], 3)
                self.assertAlmostEqual(first["exposure_seconds"], 100 * line_time)
                self.assertAlmostEqual(second["estimated_midpoint_timestamp_seconds"], 12.1 - 100 * line_time)
        words = metadata(software=3)
        words[68:] = 65535  # V3 pixels are not telemetry.
        decoded = CyperMetadataDecoder().decode_words(words)
        self.assertFalse(decoded["telemetry_available"])
        self.assertFalse(decoded["gnss_available"])
        self.assertEqual(decoded["telemetry"], [])
        self.assertEqual(decoded["imu_slot_count"], 7)

    def test_v7_keeps_seven_imu_slots_and_preserves_gnss_extension(self):
        words = metadata(software=7)
        for index in range(7):
            imu(words, index, ticks=700 + 50 * index)
        words[81:106] = np.arange(25, dtype=np.uint16) + 0x4700
        decoded = CyperMetadataDecoder().decode_words(words)
        self.assertEqual(decoded["firmware"], {"hardware": 2, "software": 7})
        self.assertEqual(decoded["imu_slot_count"], 7)
        self.assertEqual(len(decoded["imu_samples"]), 7)
        np.testing.assert_allclose([s["timestamp_seconds"] for s in decoded["imu_samples"]],
                                   12 + np.arange(700, 1001, 50) / 10000)
        self.assertEqual(decoded["raw_metadata_words"][81:106], words[81:106].tolist())
        self.assertFalse(decoded["gnss_available"])  # No decoded fix is claimed.
        with self.assertRaisesRegex(ValueError, "need 106 words"):
            CyperMetadataDecoder().decode_words(words[:105])

    def test_v7_bayer_phase_matches_v6_and_unknown_firmware_is_rejected(self):
        words = metadata(software=7)
        frame = transport(words)
        # Asymmetric Bayer samples expose an accidental red/blue phase change.
        frame[0:-1:2, 0::2, 0] = 210
        frame[1:-1:2, 1::2, 0] = 20
        frame[0:-1:2, 0::2, 1] = 30
        frame[1:-1:2, 1::2, 1] = 190
        images, firmware = decode_cyper(frame, len(words), frame.shape[0])
        self.assertEqual(firmware, (2, 7))
        frame[-1, 1, 0] = 6
        previous_images, _ = decode_cyper(frame, len(words), frame.shape[0])
        for image, previous in zip(images, previous_images):
            np.testing.assert_array_equal(image, previous)
        for unsupported in (8, 99):
            frame[-1, 1, 0] = unsupported
            with self.assertRaisesRegex(ValueError, "Unsupported/invalid"):
                decode_cyper(frame, len(words), frame.shape[0])
            with self.assertRaisesRegex(ValueError, "Unsupported/invalid"):
                CyperMetadataDecoder().decode(frame, len(words), frame.shape[0])

    def test_corrupt_or_truncated_layouts_are_rejected(self):
        for words in ([2], [1, 5] + [0] * 150, [2, 5] + [0] * 100, [2, 5, -1], [2, 5, 65536]):
            with self.subTest(words=words[:3]):
                with self.assertRaises(ValueError):
                    CyperMetadataDecoder().decode_words(words)
        with self.assertRaises(ValueError):
            metadata_words(np.zeros((8, 160, 3), np.uint8), 160, 8)


class SourceCaptureTests(unittest.TestCase):
    def test_camera_bundle_scale_is_used_and_input_override_is_respected(self):
        camera_path = Path(__file__).resolve().parents[1] / "cameras/s200032/camera.yaml"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.yaml"
            for override, expected in [(None, 9.7887), (9.75, 9.75)]:
                content = f"camera: {camera_path}\n"
                if override is not None:
                    content += f"input:\n  acceleration_scale: {override}\n"
                path.write_text(content, encoding="utf-8")
                profile, _ = load_profile(path)
                self.assertEqual(profile["input"]["acceleration_scale"], expected)
                self.assertEqual(Source(profile, {}).metadata_decoder.acceleration_scale, expected)

    def test_pair_defaults_remain_compatible(self):
        pair = Pair((None, None), 1, 2, 3)
        self.assertIsNone(pair.metadata)
        self.assertIsNone(pair.raw_transport)

    def test_source_records_metadata_before_white_balance_and_optional_raw(self):
        words = metadata()
        imu(words, 0)
        frame = transport(words)
        for preserve_raw in (False, True):
            with self.subTest(preserve_raw=preserve_raw):
                cap = MagicMock()
                cap.isOpened.return_value = True
                cap.read.return_value = (True, frame)
                cap.get.side_effect = lambda prop: 25.0 if prop == cv2.CAP_PROP_FPS else 0
                profile = {"_base": ".", "input": {"type": "cyper", "device": "/dev/fake", "width": 160,
                           "height": 8, "fps": 30, "preserve_raw": preserve_raw, "acceleration_scale": 9.7887}}
                source = Source(profile, {})
                with patch("ego2.common.sources.cv2.VideoCapture", return_value=cap), \
                     patch("ego2.common.sources.white_balance", side_effect=np.zeros_like):
                    source.open()
                    pair = source.read(threading.Event())
                    source.close()
                self.assertEqual(source.fps, 25)
                self.assertEqual(pair.metadata["source_fps"], 25)
                self.assertEqual(pair.metadata["acceleration_scale_m_s2_per_g"], 9.7887)
                self.assertEqual(pair.metadata["raw_metadata_words"], words.tolist())
                self.assertEqual(pair.metadata["host_receive_monotonic_seconds"], pair.stamp)
                self.assertEqual(pair.metadata["host_decode_complete_monotonic_seconds"], pair.received)
                self.assertGreaterEqual(pair.received, pair.stamp)
                self.assertTrue(all(not image.any() for image in pair.images))
                if preserve_raw:
                    np.testing.assert_array_equal(pair.raw_transport, frame)
                    self.assertFalse(np.shares_memory(pair.raw_transport, frame))
                else:
                    self.assertIsNone(pair.raw_transport)


if __name__ == "__main__":
    unittest.main()
