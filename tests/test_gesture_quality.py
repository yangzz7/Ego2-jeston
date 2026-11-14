"""Sensor continuity evidence stays separate from host stalls and empty slots."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ego2.gesture.quality import CaptureQuality, audit_dataset


def sample(stamp, *, accepted=True, empty=False):
    return dict(timestamp_seconds=stamp, accepted=accepted,
                raw_words=[0] * 9 if empty else [1] * 9)


def row(sequence, device, imu=(), *, host=None):
    metadata = dict(source_fps=30.0, image_timestamp_seconds=device,
                    image_timestamp_fields_in_range=True,
                    host_receive_monotonic_seconds=host if host is not None else sequence / 30,
                    imu_slots=list(imu))
    return dict(sequence=sequence, metadata=metadata, receive_timestamp=sequence / 30 + 10)


class QualityTests(unittest.TestCase):
    def test_quantized_camera_clock_and_six_or_seven_imu_slots_are_continuous(self):
        quality = CaptureQuality()
        first = row(1, 50, [sample(10 + index * .005) for index in range(7)])
        second = row(2, 50.0333, [sample(10 + index * .005) for index in range(7, 13)]
                     + [sample(0, accepted=False, empty=True)])
        original = copy.deepcopy(second)
        self.assertEqual(quality.observe(first), [])
        self.assertEqual(quality.observe(second), [])
        summary = quality.summary()
        self.assertEqual(summary["device_rgb"]["samples"], 2)
        self.assertEqual(summary["imu"]["samples"], 13)
        self.assertEqual(summary["imu"]["ignored_zero_slots"], 1)
        self.assertEqual(summary["imu"]["ignored_invalid_slots"], 1)
        self.assertEqual(summary["imu"]["gap_count"], 0)
        self.assertEqual(second, original)

    def test_detects_known_large_rgb_and_imu_intervals(self):
        quality = CaptureQuality()
        quality.observe(row(902, 347207.6159, [sample(347207.6412)], host=100))
        events = quality.observe(row(903, 347208.2493, [sample(347208.2462)], host=100.0333))
        summary = quality.summary()
        self.assertEqual([event["stream"] for event in events], ["device_rgb", "imu"])
        self.assertEqual(summary["device_rgb"]["estimated_missing_samples"], 18)
        self.assertEqual(summary["imu"]["estimated_missing_samples"], 120)
        self.assertAlmostEqual(summary["device_rgb"]["max_interval_seconds"], .6334)
        self.assertAlmostEqual(summary["imu"]["max_interval_seconds"], .605)
        self.assertEqual(summary["device_rgb"]["event_examples"][0]["previous_sequence"], 902)

    def test_host_stall_does_not_become_a_sensor_gap(self):
        quality = CaptureQuality()
        quality.observe(row(1, 0, [sample(0)], host=100))
        events = quality.observe(row(2, .0333, [sample(.005)], host=100.7))
        self.assertEqual([event["stream"] for event in events], ["host_receive"])
        result = quality.summary()
        self.assertEqual(result["host_receive"]["gap_count"], 1)
        self.assertIsNone(result["host_receive"]["estimated_missing_samples"])
        self.assertEqual(result["device_rgb"]["estimated_missing_samples"], 0)
        self.assertEqual(result["imu"]["estimated_missing_samples"], 0)

    def test_overlapping_imu_packets_deduplicate_without_moving_clock_backwards(self):
        quality = CaptureQuality()
        quality.observe(row(1, 0, [sample(10), sample(10.005), sample(10.01)]))
        quality.observe(row(2, .0333, [sample(10.005), sample(10.01), sample(10.015)]))
        quality.observe(row(3, .0667, [sample(9.99), sample(10.02)]))
        result = quality.summary()["imu"]
        self.assertEqual(result["accepted_observations"], 8)
        self.assertEqual(result["samples"], 5)
        self.assertEqual(result["duplicate_timestamps"], 2)
        self.assertEqual(result["out_of_order_timestamps"], 1)
        self.assertEqual(result["gap_count"], 0)

    def test_missing_sensor_metadata_is_not_substituted_from_host_clock(self):
        quality = CaptureQuality(fps=30)
        quality.observe(dict(sequence=1, receive_timestamp=100, metadata={}))
        quality.observe(dict(sequence=2, receive_timestamp=101, metadata={}))
        result = quality.summary()
        self.assertFalse(result["device_rgb"]["available"])
        self.assertFalse(result["imu"]["available"])
        self.assertEqual(result["device_rgb"]["missing_timestamps"], 2)
        self.assertEqual(result["host_receive"]["gap_count"], 1)

    def test_invalid_timestamps_and_zero_slots_do_not_create_spurious_sensor_gaps(self):
        quality = CaptureQuality()
        first = row(1, float("nan"), [sample(float("inf")), sample(0, empty=True),
                                      sample(-900, accepted=False)])
        quality.observe(first)
        quality.observe(row(2, 10, [sample(20)]))
        result = quality.summary()
        self.assertEqual(result["device_rgb"]["invalid_timestamps"], 1)
        self.assertEqual(result["imu"]["invalid_timestamps"], 1)
        self.assertEqual(result["imu"]["ignored_invalid_slots"], 2)
        self.assertEqual(result["imu"]["gap_count"], 0)
        json.dumps(result, allow_nan=False)

    def test_device_restart_resets_both_device_timelines(self):
        quality = CaptureQuality()
        quality.observe(row(1, 500, [sample(500)], host=100))
        events = quality.observe(row(2, 1, [sample(1)], host=100.0333))
        quality.observe(row(3, 1.0333, [sample(1.005)], host=100.0667))
        self.assertEqual(events[0]["kind"], "timestamp_regression")
        result = quality.summary()
        self.assertEqual(result["device_rgb"]["clock_resets"], 1)
        self.assertEqual(result["imu"]["clock_resets"], 1)
        self.assertEqual(result["imu"]["samples"], 3)
        self.assertEqual(result["imu"]["gap_count"], 0)

    def test_examples_are_bounded_but_every_gap_is_counted(self):
        quality = CaptureQuality(max_events=3)
        for index in range(100):
            quality.observe(row(index, index * .1, [sample(index * .01)]))
        result = quality.summary()
        self.assertEqual(result["device_rgb"]["gap_count"], 99)
        self.assertEqual(result["imu"]["gap_count"], 99)
        self.assertEqual(len(result["device_rgb"]["event_examples"]), 3)
        result["device_rgb"]["event_examples"][0]["sequence"] = -100
        self.assertNotEqual(quality.summary()["device_rgb"]["event_examples"][0]["sequence"], -100)

    def test_streamed_journal_audit_is_read_only_and_reports_malformed_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "raw/capture.jsonl"
            path.parent.mkdir()
            raw = "".join(json.dumps(value) + "\n" for value in [
                row(1, 1, [sample(2)]), row(2, 1.0667, [sample(2.04)])])
            path.write_text(raw)
            result = audit_dataset(root)
            self.assertEqual(result["capture_rows"], 2)
            self.assertEqual(result["device_rgb"]["estimated_missing_samples"], 1)
            self.assertEqual(result["imu"]["estimated_missing_samples"], 7)
            self.assertEqual(path.read_text(), raw)
            self.assertEqual([p for p in root.rglob("*") if p.is_file()], [path])
            path.write_text(raw + "broken\n")
            with self.assertRaisesRegex(ValueError, r"capture.jsonl:3"):
                audit_dataset(path)


if __name__ == "__main__":
    unittest.main()
