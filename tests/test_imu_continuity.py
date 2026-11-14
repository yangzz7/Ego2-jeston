"""Regression tests for preview-only discontinuities and stage attribution."""
import copy
import io
import json
import queue
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock

import numpy as np

from ego2.multi_vio.dashboard_transport import DashboardPublisher
from ego2.multi_vio.orientation_service import AttitudeInput
from ego2.multi_vio.raw_imu import RawImuLogger
from tests.test_raw_imu_preview import pair


def rows(start=1., count=20):
    stamps = start+np.arange(count)*.005
    values = np.arange(count*6, dtype=float).reshape(count,6)+.25
    return np.column_stack((stamps, stamps, values))


class SeparatorTests(unittest.TestCase):
    def setUp(self):
        with redirect_stdout(io.StringIO()):
            self.publisher = DashboardPublisher(queue.Queue(), 'head', 0., {})

    def append(self, data, dropped=0, epoch=0):
        before = data.copy()
        self.publisher.append_raw(data, dict(input_dropped=dropped, clock_epoch=epoch))
        np.testing.assert_array_equal(data, before)

    def raw(self):
        return np.asarray(self.publisher.raw.rows)

    def assert_breaks(self, count):
        self.assertEqual(np.isnan(self.raw()[:,2]).sum(), count)

    def test_continuous_batches_have_no_breaks(self):
        first, second = rows(), rows(1.1)
        self.append(first); self.append(second)
        self.assert_breaks(0)
        np.testing.assert_array_equal(self.raw(), np.concatenate((first, second)))

    def test_drop_counter_jump_breaks_only_once_not_every_sample(self):
        first, second, third = rows(), rows(1.1), rows(1.2)
        self.append(first); self.append(second, dropped=9); self.append(third, dropped=9)
        self.assert_breaks(1)
        raw = self.raw()
        np.testing.assert_array_equal(raw[np.isfinite(raw[:,2])], np.concatenate((first, second, third)))
        self.assertEqual(self.publisher.raw_stats['input_loss_events'], 1)
        self.assertEqual(self.publisher.raw_stats['input_loss_breaks'], 1)
        marker = raw[np.isnan(raw[:,2]),0][0]
        self.assertLess(round(marker*1e9), round(second[0,0]*1e9))

    def test_two_distinct_losses_have_two_boundaries(self):
        self.append(rows()); self.append(rows(1.1), 1); self.append(rows(1.2), 2)
        self.assert_breaks(2)

    def test_existing_overflow_marker_is_not_duplicated(self):
        first, second = rows(), rows(1.2)
        marker = np.array([[1.19, np.nan, *([np.nan]*6)]])
        self.append(first); self.append(np.concatenate((marker, second)), dropped=20)
        self.assert_breaks(1)
        self.assertEqual(self.publisher.raw_stats['inserted_breaks'], 0)
        self.assertEqual(self.publisher.raw_stats['upstream_breaks'], 1)
        np.testing.assert_array_equal(self.raw()[-len(second):], second)

    def test_existing_marker_survives_packet_boundary_without_extra_break(self):
        self.append(rows())
        self.append(np.array([[1.1, np.nan, *([np.nan]*6)]]), dropped=1)
        self.append(rows(1.4), dropped=1)
        self.assert_breaks(1)

    def test_consecutive_markers_deduplicated_without_losing_valid_values(self):
        self.append(rows())
        self.append(np.array([[1.1, np.nan, *([np.nan]*6)], [1.2, np.nan, *([np.nan]*6)]]))
        self.append(rows(1.3))
        self.assert_breaks(1)
        self.assertEqual(self.publisher.raw_stats['duplicate_markers_suppressed'], 1)

    def test_real_gaps_inside_batch_remain_separate(self):
        data = np.concatenate((rows(count=3), rows(1.08,count=3), rows(1.23,count=3)))
        self.append(data)
        self.assert_breaks(2)
        raw = self.raw()
        np.testing.assert_array_equal(raw[np.isfinite(raw[:,2])], data)
        self.assertEqual(self.publisher.input_continuity.stats['gaps'], 2)
        self.assertAlmostEqual(self.publisher.input_continuity.stats['max_gap_ms'], 140.)

    def test_same_boundary_loss_and_time_gap_need_only_one_marker(self):
        self.append(rows()); self.append(rows(1.2), dropped=1)
        self.assert_breaks(1)

    def test_empty_batch_retains_pending_loss_until_a_sample_arrives(self):
        self.append(rows()); self.append(np.empty((0,8)), dropped=1)
        self.append(rows(1.1), dropped=1)
        self.assert_breaks(1)

    def test_clock_epoch_change_breaks_once_and_does_not_count_clock_jump_as_gap(self):
        self.append(rows())
        data=rows(.1); data[:,0]+=1.1
        self.append(data, epoch=1)
        self.assert_breaks(1)
        self.assertEqual(self.publisher.input_continuity.stats['gaps'], 0)
        self.assertEqual(self.publisher.input_continuity.stats['epochs'], 1)
        np.testing.assert_array_equal(self.raw()[-len(data):], data)

    def test_input_counter_reset_is_not_a_new_loss(self):
        self.append(rows(), dropped=5); self.append(rows(1.1), dropped=0)
        self.assert_breaks(1)  # Initial known loss only.


class AttributionTests(unittest.TestCase):
    def publisher(self, q):
        alive=threading.Event(); alive.set()
        return AttitudeInput(q, 'head', 100., alive)

    def test_source_gap_and_host_delay_are_measured_separately(self):
        q=queue.Queue(); p=self.publisher(q)
        p.capture(pair()); q.get()
        p.capture(pair(image=12.115, received=105.2))
        info=q.get()['raw_source_diagnostics']
        self.assertEqual(info['decoded']['gaps'], 0)
        self.assertEqual(info['before_queue']['gaps'], 0)
        self.assertAlmostEqual(info['capture']['max_receipt_interval_ms'], 200.)
        p.capture(pair(image=12.265, received=105.3))
        info=q.get()['raw_source_diagnostics']
        self.assertEqual(info['decoded']['gaps'], 1)
        self.assertEqual(info['before_queue']['gaps'], 1)
        self.assertAlmostEqual(info['decoded']['max_gap_ms'], 140.)

    def test_queue_loss_is_distinct_from_source_gaps(self):
        q=queue.Queue(maxsize=1); q.put('busy'); p=self.publisher(q)
        for i in range(30):
            p.capture(pair(image=12.1+i*.015, received=105.+i*.015))
        self.assertGreater(p.stats['dropped'], 0)
        q.get(); p.flush(); packet=q.get()
        info=packet['raw_source_diagnostics']
        self.assertEqual(info['decoded']['gaps'], 0)
        self.assertEqual(info['before_queue']['samples'], 90)
        self.assertGreater(info['input_queue']['queue_full'], 0)
        with redirect_stdout(io.StringIO()):
            out=DashboardPublisher(queue.Queue(), 'head', 100., {})
        out.append_raw(packet['imu'],packet)
        self.assertLess(out.input_continuity.stats['samples'], 90)
        self.assertEqual(out.raw_stats['inserted_breaks'], 0)
        self.assertEqual(out.raw_stats['upstream_breaks'], 1)

    def test_decoder_rejection_reasons_and_metadata_remain_intact(self):
        q=queue.Queue(); p=self.publisher(q); item=pair()
        # Accepted decoder values are unchanged; inspect rejected slots only.
        item.metadata['imu_slots'].append(dict(accepted=False,rejection_reason='temperature_discontinuity'))
        original=copy.deepcopy(item.metadata)
        p.capture(item); packet=q.get()
        self.assertEqual(item.metadata, original)
        self.assertEqual(packet['raw_source_diagnostics']['rejection_reasons']['temperature_discontinuity'], 1)
        json.dumps(packet['raw_source_diagnostics'],allow_nan=False)

    def test_plot_counters_only_advance_after_both_channels_submitted(self):
        rec=MagicMock(); logger=RawImuLogger(rec,['head'])
        data=rows(); packet=dict(role='head',imu=data,raw_diagnostics={'test':1})
        rec.send_columns.side_effect=[None, RuntimeError('viewer unavailable')]
        with self.assertRaises(RuntimeError): logger.consume(packet)
        self.assertEqual(logger.metrics()['head']['viewer_submitted']['samples'],0)
        rec.send_columns.side_effect=None
        logger.consume(packet)
        self.assertEqual(logger.metrics()['head']['viewer_submitted']['samples'],20)
        self.assertEqual(logger.metrics()['head']['test'],1)


if __name__ == '__main__':
    unittest.main()
