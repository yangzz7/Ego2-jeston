"""Read-only, bounded IMU continuity counters. Never modify sensor samples."""
import math
from collections import Counter

# Preserve the dashboard's existing discontinuity policy (nominal 5 ms IMU).
RAW_GAP_SECONDS = .0151


class Continuity:
    def __init__(self):
        self.last = None
        self.epoch = None
        self.stats = dict(samples=0, markers=0, gaps=0, max_gap_ms=0.,
                          repeat_or_old=0, epochs=0)

    def set_epoch(self, epoch):
        if self.epoch is not None and epoch != self.epoch:
            self.last = None
            self.stats['epochs'] += 1
        self.epoch = epoch

    def sample(self, stamp):
        try:
            stamp = float(stamp)
        except (TypeError, ValueError, OverflowError):
            stamp = math.nan
        if not math.isfinite(stamp):
            self.stats['markers'] += 1
            return
        if self.last is not None:
            delta = stamp-self.last
            if delta <= 0:
                self.stats['repeat_or_old'] += 1
                return
            self.stats['max_gap_ms'] = max(self.stats['max_gap_ms'], delta*1000)
            if delta > RAW_GAP_SECONDS:
                self.stats['gaps'] += 1
        self.last = stamp
        self.stats['samples'] += 1

    def row(self, row):
        if all(math.isfinite(float(v)) for v in row[1:]):
            self.sample(row[1])  # Device time, not callback/send completion time.
        else:
            self.stats['markers'] += 1

    def snapshot(self):
        return dict(self.stats)


class CaptureDiagnostics:
    """Inspect existing decoded metadata; no extra USB reads or decoder changes."""
    def __init__(self):
        self.decoded = Continuity()
        self.pre_queue = Continuity()
        self.rejections = Counter()
        self.epoch = 0
        self.last_image = self.last_received = None
        self.stats = dict(frames=0, slots=0, rejected_slots=0,
                          max_image_interval_ms=0., max_receipt_interval_ms=0.)

    def capture(self, pair):
        meta = pair.metadata or {}
        self.stats['frames'] += 1
        image = meta.get('image_timestamp_seconds')
        if image is not None and math.isfinite(float(image)):
            image = float(image)
            if self.last_image is not None:
                if image < self.last_image:
                    self.epoch += 1
                else:
                    self.stats['max_image_interval_ms'] = max(
                        self.stats['max_image_interval_ms'], (image-self.last_image)*1000)
            self.last_image = image
        self.decoded.set_epoch(self.epoch)
        received = float(pair.received)
        if math.isfinite(received):
            if self.last_received is not None:
                self.stats['max_receipt_interval_ms'] = max(
                    self.stats['max_receipt_interval_ms'], (received-self.last_received)*1000)
            self.last_received = received
        slots = meta.get('imu_slots', ())
        self.stats['slots'] += len(slots)
        for sample in slots:
            if sample.get('accepted') is False:
                self.stats['rejected_slots'] += 1
                self.rejections[str(sample.get('rejection_reason') or 'unspecified')] += 1
        for sample in meta.get('imu_samples', ()):
            if sample.get('accepted', True):
                self.decoded.sample(sample.get('timestamp_seconds', math.nan))

    def snapshot(self, publisher):
        return dict(capture=dict(self.stats), decoded=self.decoded.snapshot(),
                    before_queue=self.pre_queue.snapshot(), rejection_reasons=dict(self.rejections),
                    input_queue={key:publisher.stats.get(key, 0) for key in
                                 ('samples','invalid','duplicates','missing_frames','resets',
                                  'dropped','queue_full','failures','submit_ms_max')})
