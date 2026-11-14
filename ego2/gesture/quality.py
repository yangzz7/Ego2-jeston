"""Read-only, bounded-memory continuity checks for recorded capture timestamps.

Device RGB/IMU clocks determine sensor gaps. Host receive delays are reported
separately and never counted as missing sensor frames. Nothing is interpolated.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _positive(value, name):
    result = _number(value)
    if result is None or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


class _Timeline:
    def __init__(self, name, hz, max_events, *, estimate_missing=True, deduplicate=False):
        self.name, self.hz = name, hz
        self.max_events = max_events
        self.estimate_missing, self.deduplicate = estimate_missing, deduplicate
        self.samples = self.missing_timestamps = self.invalid_timestamps = 0
        self.duplicates = self.out_of_order = self.resets = 0
        self.gap_count = self.estimated_missing = 0
        self.max_interval = 0.0
        self.last = self.last_sequence = None
        self.examples = []
        # Typical firmware overlap is one adjacent image packet. Retain a much
        # longer window while keeping memory independent of recording duration.
        self.recent = deque(maxlen=4096)
        self.seen = set()

    def _remember(self, value):
        if len(self.recent) == self.recent.maxlen:
            self.seen.remove(self.recent[0])
        self.recent.append(value)
        self.seen.add(value)

    def reset(self):
        self.last = self.last_sequence = None
        self.recent.clear()
        self.seen.clear()
        self.resets += 1

    def observe(self, value, sequence, *, valid=True):
        if value is None:
            self.missing_timestamps += 1
            return None
        stamp = _number(value)
        if not valid or stamp is None:
            self.invalid_timestamps += 1
            return None
        if stamp in self.seen or stamp == self.last:
            self.duplicates += 1
            return None
        if self.last is not None and stamp < self.last:
            self.out_of_order += 1
            if self.deduplicate:
                # Older/overlapping IMU samples never move the high-water mark
                # backwards and therefore cannot create a false future gap.
                return None
            event = dict(stream=self.name, kind="timestamp_regression", sequence=sequence,
                         previous_sequence=self.last_sequence, timestamp_seconds=stamp,
                         previous_timestamp_seconds=self.last)
            if len(self.examples) < self.max_events:
                self.examples.append(event)
            self.reset()
        else:
            event = None
        if self.last is not None:
            interval = stamp - self.last
            self.max_interval = max(self.max_interval, interval)
            # Timestamps have 0.1 ms quantization on current Cyper firmware.
            # A 1.5-period threshold tolerates this without masking a whole
            # missing sample. Estimates use the nominal hardware rate.
            if self.hz is not None and interval > 1.5 / self.hz:
                missing = max(0, math.floor(interval * self.hz + 0.5) - 1)
                self.gap_count += 1
                self.estimated_missing += missing if self.estimate_missing else 0
                event = dict(stream=self.name, kind="timestamp_gap", sequence=sequence,
                             previous_sequence=self.last_sequence,
                             previous_timestamp_seconds=self.last, timestamp_seconds=stamp,
                             interval_seconds=interval,
                             expected_interval_seconds=1 / self.hz)
                if self.estimate_missing:
                    event["estimated_missing_samples"] = missing
                if len(self.examples) < self.max_events:
                    self.examples.append(event)
        self.samples += 1
        self.last, self.last_sequence = stamp, sequence
        if self.deduplicate:
            self._remember(stamp)
        return event

    def summary(self):
        return dict(available=self.samples > 0, expected_hz=self.hz, samples=self.samples,
                    missing_timestamps=self.missing_timestamps, invalid_timestamps=self.invalid_timestamps,
                    duplicate_timestamps=self.duplicates, out_of_order_timestamps=self.out_of_order,
                    clock_resets=self.resets, gap_count=self.gap_count,
                    estimated_missing_samples=self.estimated_missing if self.estimate_missing else None,
                    max_interval_seconds=self.max_interval,
                    event_examples=[dict(event) for event in self.examples],
                    event_examples_limit=self.max_events)


class CaptureQuality:
    """Observe capture journal rows without retaining images or full journals.

    ``fps`` overrides the saved ``metadata.source_fps``. IMU defaults to the
    current Cyper 200 Hz cadence; callers with another sensor must supply its
    actual rate. Gap counts are timestamp evidence, not a diagnosis of cause.
    """
    def __init__(self, fps=None, *, imu_hz=200.0, max_events=20):
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 0:
            raise ValueError("max_events must be a nonnegative integer")
        self.fixed_fps = None if fps is None else _positive(fps, "fps")
        self.rgb = _Timeline("device_rgb", self.fixed_fps, max_events)
        self.imu = _Timeline("imu", _positive(imu_hz, "imu_hz"), max_events, deduplicate=True)
        self.host = _Timeline("host_receive", self.fixed_fps, max_events, estimate_missing=False)
        self.capture_rows = self.fps_changes = 0
        self.imu_accepted_observations = self.imu_invalid_slots = self.imu_zero_slots = 0
        self.rows_without_imu = 0

    def observe(self, row):
        """Return this row's detected timing events; never alter the input row."""
        if not isinstance(row, dict) or not isinstance(row.get("metadata", {}), dict):
            raise ValueError("Capture quality expects a capture journal row with metadata mapping")
        metadata = row.get("metadata", {})
        sequence = row.get("sequence")
        self.capture_rows += 1
        if self.fixed_fps is None:
            fps = _number(metadata.get("source_fps"))
            if fps is not None and fps > 0:
                if self.rgb.hz is not None and not math.isclose(fps, self.rgb.hz, rel_tol=1e-6):
                    self.fps_changes += 1
                self.rgb.hz = self.host.hz = fps
        events = []
        rgb_event = self.rgb.observe(metadata.get("image_timestamp_seconds"), sequence,
            valid=metadata.get("image_timestamp_fields_in_range", True) is not False)
        if rgb_event:
            events.append(rgb_event)
            if rgb_event["kind"] == "timestamp_regression":
                # A device restart affects both device clocks. Do not compare
                # new IMU counters to the previous boot's high-water mark.
                self.imu.reset()
        host_stamp = metadata.get("host_receive_monotonic_seconds", row.get("receive_timestamp"))
        host_event = self.host.observe(host_stamp, sequence)
        if host_event:
            events.append(host_event)
        # imu_samples is already filtered by the SDK heuristic. Prefer raw
        # slots when available so we can also count rejected/zero padding.
        slots = metadata.get("imu_slots")
        accepted_only = slots is None
        if accepted_only:
            slots = metadata.get("imu_samples", [])
        if not isinstance(slots, list):
            raise ValueError("Capture metadata IMU slots/samples must be a list")
        accepted = []
        for sample in slots:
            if not isinstance(sample, dict):
                raise ValueError("Capture metadata IMU sample must be a mapping")
            words = sample.get("raw_words")
            zero = isinstance(words, list) and bool(words) and all(value == 0 for value in words)
            if zero:
                self.imu_zero_slots += 1
            if zero or not sample.get("accepted", accepted_only):
                self.imu_invalid_slots += 1
                continue
            self.imu_accepted_observations += 1
            accepted.append(sample)
        if not accepted:
            self.rows_without_imu += 1
        # Preserve stored order; unexpected reordering is itself diagnostic.
        for sample in accepted:
            event = self.imu.observe(sample.get("timestamp_seconds"), sequence)
            if event:
                events.append(event)
        return events

    def summary(self):
        """Return a JSON-safe snapshot; examples remain bounded by max_events."""
        rgb, imu, host = self.rgb.summary(), self.imu.summary(), self.host.summary()
        rgb["timestamp_field"] = "metadata.image_timestamp_seconds"
        imu.update(timestamp_field="metadata.imu_slots/imu_samples[].timestamp_seconds",
                   accepted_observations=self.imu_accepted_observations,
                   ignored_invalid_slots=self.imu_invalid_slots,
                   ignored_zero_slots=self.imu_zero_slots,
                   rows_without_accepted_imu=self.rows_without_imu,
                   deduplication="Repeated or older timestamps do not advance the timeline; a 4096-sample window identifies repeats.")
        host["timestamp_field"] = "metadata.host_receive_monotonic_seconds, falling back to receive_timestamp"
        return dict(schema_version=1, capture_rows=self.capture_rows,
                    nominal_fps_changes=self.fps_changes, gap_threshold_periods=1.5,
                    device_rgb=rgb, imu=imu, host_receive=host,
                    semantics="Device gaps estimate absent samples from nominal cadence; host delays alone do not establish sensor loss. No data are inserted or modified.")


def audit_dataset(root, *, fps=None, imu_hz=200.0, max_events=20):
    """Stream a dataset's raw/capture.jsonl, or an explicitly supplied journal."""
    path = Path(root).expanduser()
    if path.is_dir():
        path = path / "raw/capture.jsonl"
    quality = CaptureQuality(fps, imu_hz=imu_hz, max_events=max_events)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                quality.observe(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid capture journal {path}:{line_number}: {exc}") from exc
    return quality.summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="dataset directory or raw/capture.jsonl; read-only")
    parser.add_argument("--fps", type=float, help="expected RGB rate; otherwise use saved source_fps")
    parser.add_argument("--imu-hz", type=float, default=200.0, help="expected IMU rate (default: 200 Hz)")
    parser.add_argument("--max-events", type=int, default=20, help="maximum examples per timeline")
    args = parser.parse_args(argv)
    try:
        result = audit_dataset(args.dataset, fps=args.fps, imu_hz=args.imu_hz, max_events=args.max_events)
    except (OSError, ValueError) as exc:
        print(f"[quality] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
