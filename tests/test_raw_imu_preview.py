"""Raw sensor integrity, clocks, backpressure, capture isolation and UI layout."""
import copy
import io
import queue
import tempfile
import threading
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import rerun as rr
import rerun.blueprint as b

from ego2.common.sources import Pair
from ego2.common.runtime import LatestPairs
from ego2.common.cyper_metadata import CyperMetadataDecoder
from ego2.multi_vio.raw_imu import RawImuPublisher, RawImuLogger, FIELDS
from ego2.multi_vio.mode_viewer import BatchRecording, GestureSink, build_blueprint
from tests.test_cyper_metadata import metadata, imu


def pair(image=12.1, received=105., count=3, seq=1):
    words = metadata(seconds=int(image), ticks=round(image % 1 * 10000))
    for i in range(count):
        imu(words, i, seconds=int(image), ticks=round(image % 1 * 10000) - (count - i - 1)*50)
    decoded = CyperMetadataDecoder().decode_words(words)
    return Pair((np.zeros((8, 8, 3), np.uint8),)*2, received, received, seq, decoded)


class ImuPreviewTests(unittest.TestCase):
    def test_raw_values_and_sample_offsets_exact_with_input_unchanged(self):
        q = queue.Queue()
        publisher = RawImuPublisher(q, 'head', 100.)
        p = pair()
        before = copy.deepcopy(p.metadata)
        publisher.capture(p)
        rows = q.get_nowait()['imu']
        expected = np.array([s[FIELDS[0]] + s[FIELDS[1]] for s in p.metadata['imu_samples']])
        np.testing.assert_array_equal(rows[:, 2:], expected)
        np.testing.assert_allclose(rows[:, 0], [4.99, 4.995, 5.])
        np.testing.assert_array_equal(rows[:, 1], [s['timestamp_seconds'] for s in p.metadata['imu_samples']])
        self.assertEqual(p.metadata, before)
        np.testing.assert_array_equal(p.images[0], 0)

    def test_duplicate_samples_are_not_repeated(self):
        q = queue.Queue()
        publisher = RawImuPublisher(q, 'chest', 100.)
        publisher.capture(pair())
        publisher.capture(pair(received=105.2))
        self.assertEqual(q.qsize(), 1)
        self.assertEqual(publisher.stats['duplicates'], 3)

    def test_device_sample_intervals_survive_usb_receipt_jitter(self):
        q = queue.Queue()
        publisher = RawImuPublisher(q, 'head', 100.)
        publisher.capture(pair())
        first = q.get_nowait()['imu']
        # Next sensor frame is 15 ms later but delayed in USB by 200 ms.
        publisher.capture(pair(image=12.115, received=105.2))
        second = q.get_nowait()['imu']
        np.testing.assert_allclose(np.diff(np.r_[first[:, 0], second[:, 0]]), .005, atol=1e-12)

    def test_missing_samples_create_a_gap_not_zero_values(self):
        q = queue.Queue()
        publisher = RawImuPublisher(q, 'head', 100.)
        with redirect_stdout(io.StringIO()):
            publisher.capture(pair(count=0))
            publisher.capture(pair(count=0, received=105.2))
        self.assertEqual(q.qsize(), 1)
        self.assertTrue(np.isnan(q.get_nowait()['imu'][:, 2:]).all())
        self.assertEqual(publisher.stats['samples'], 0)

    def test_rejected_and_nonfinite_values_do_not_enter_waveforms(self):
        q = queue.Queue()
        publisher = RawImuPublisher(q, 'head', 100.)
        p = pair()
        p.metadata['imu_samples'][0]['accepted'] = False
        p.metadata['imu_samples'][1][FIELDS[0]][1] = float('inf')
        publisher.capture(p)
        rows = q.get_nowait()['imu']
        self.assertEqual(len(rows), 1)
        self.assertEqual(publisher.stats['invalid'], 2)

    def test_device_reset_accepts_new_epoch_and_breaks_lines(self):
        q = queue.Queue()
        publisher = RawImuPublisher(q, 'head', 100.)
        publisher.capture(pair())
        publisher.capture(pair(image=1.1, received=105.2))
        q.get_nowait()
        rows = q.get_nowait()['imu']
        self.assertEqual(publisher.stats['resets'], 1)
        self.assertEqual(np.isfinite(rows[:, 2]).sum(), 3)
        self.assertEqual(np.isnan(rows[:, 2]).sum(), 1)
        self.assertTrue((np.diff(rows[:, 0]) > 0).all())

    def test_queue_backpressure_retains_bounded_recent_samples(self):
        q = queue.Queue(maxsize=1)
        publisher = RawImuPublisher(q, 'head', 100., capacity=8)
        q.put('blocked')
        for i in range(12):
            publisher.capture(pair(image=12.1+i*.01, received=105.+i*.01))
        self.assertEqual(len(publisher.rows), 8)
        self.assertGreater(publisher.stats['dropped'], 0)
        q.get_nowait()
        publisher.flush()
        rows = q.get_nowait()['imu']
        self.assertEqual(len(rows), 8)
        self.assertTrue(np.isnan(rows[0, 2:]).all())
        self.assertEqual(publisher.stats['failures'], 0)

    def test_capture_observes_every_pair_despite_latest_inference_slot(self):
        pending = queue.Queue()
        publisher = RawImuPublisher(pending, 'head', 100., interval=.01)
        samples = [pair(image=12.1+i*.02, received=105.+i*.02, seq=i) for i in range(15)]
        iterator = iter(samples)
        archived = []
        source = SimpleNamespace(open=lambda: None, close=lambda: None,
                                 read=lambda stop: next(iterator, None))
        def observe(p):
            archived.append(p)
            publisher.capture(p)
        capture = LatestPairs(source, threading.Event(), on_pair=observe).start()
        capture.thread.join(5)
        self.assertFalse(capture.thread.is_alive())
        self.assertEqual(capture.dropped, 14)
        self.assertIs(capture.read(), samples[-1])
        capture.close()
        self.assertEqual(archived, samples)
        rows = np.concatenate([pending.get_nowait()['imu'] for _ in range(pending.qsize())])
        self.assertEqual(len(rows), 45)

    def test_preview_failure_never_escapes_to_capture(self):
        q = MagicMock()
        q.put_nowait.side_effect = OSError('closed viewer queue')
        publisher = RawImuPublisher(q, 'head', 100.)
        with redirect_stderr(io.StringIO()):
            publisher.capture(pair())
            publisher.capture(pair(image=12.2, received=105.2))
        self.assertTrue(publisher.disabled)
        self.assertEqual(publisher.stats['failures'], 1)

    def test_imu_queue_is_independent_of_full_image_queue(self):
        image_queue, imu_queue = queue.Queue(maxsize=1), queue.Queue()
        image_queue.put('image busy')
        bridge = BatchRecording(image_queue, 'head', 100., imu_queue=imu_queue)
        sink = GestureSink({'viewer': {'jpeg_quality': 85}}, bridge)
        sink.capture_imu(pair())
        self.assertEqual(imu_queue.qsize(), 1)
        self.assertEqual(image_queue.get_nowait(), 'image busy')
        self.assertEqual(bridge.imu_stats['samples'], 3)

    def test_logger_preserves_all_six_channels_and_does_not_set_image_time(self):
        q = queue.Queue()
        RawImuPublisher(q, 'head', 100.).capture(pair())
        packet = q.get_nowait()
        recording = MagicMock()
        logger = RawImuLogger(recording, ['head', 'chest'])
        logger.consume(packet)
        self.assertEqual(recording.send_columns.call_count, 2)
        self.assertTrue(all(c.args[0].startswith('/imu/head/') for c in recording.send_columns.call_args_list))
        for group, call in enumerate(recording.send_columns.call_args_list):
            values = next(iter(call.kwargs['columns'])).as_arrow_array().to_pylist()
            np.testing.assert_array_equal(values, packet['imu'][:, 2+group*3:5+group*3])
        recording.set_time.assert_not_called()
        recording.reset_time.assert_not_called()
        with self.assertRaises(ValueError):
            logger.consume(dict(packet, role='not_selected'))
        # Also serialize through the real SDK to catch invalid Arrow batches.
        with tempfile.TemporaryDirectory() as tmp:
            rec = rr.RecordingStream('raw-imu-test')
            rec.save(Path(tmp)/'sample.rrd')
            RawImuLogger(rec, ['head']).consume(packet)
            rec.flush(); rec.disconnect()
            self.assertGreater((Path(tmp)/'sample.rrd').stat().st_size, 0)


class ImuLayoutTests(unittest.TestCase):
    def cfg(self, roles, mode='gesture'):
        return dict(mode=mode, viewer=dict(row_order=roles),
                    devices={role: dict(sn=f's20000{i}') for i, role in enumerate(roles)})

    def test_single_has_two_images_above_quaternion_and_status(self):
        root = build_blueprint(self.cfg(['chest'])).root_container
        self.assertEqual(len(root.contents), 2)
        self.assertTrue(all(isinstance(v, b.Spatial2DView) for v in root.contents[0].contents))
        self.assertEqual([type(v) for v in root.contents[1].contents[0].contents],
                         [b.TimeSeriesView]*3+[b.Spatial3DView])
        self.assertIsInstance(root.contents[1].contents[1], b.TextDocumentView)
        self.assertEqual(len(root.contents[1].contents), 2)

    def test_four_rows_each_have_two_images_and_own_quaternion_status(self):
        roles = ['head', 'chest', 'left_hand', 'right_hand']
        blueprint = build_blueprint(self.cfg(roles))
        self.assertEqual(len(blueprint.root_container.contents), 4)
        for role, row in zip(roles, blueprint.root_container.contents):
            self.assertEqual(len(row.contents), 3)
            self.assertTrue(all(isinstance(v, b.Spatial2DView) for v in row.contents[:2]))
            plots = row.contents[2].contents
            self.assertEqual(len(plots), 2)
            self.assertTrue(all(v.origin.startswith(f'/imu/{role}/') for v in plots[0].contents[:3]))
            self.assertEqual(plots[0].contents[3].origin, f'/body/{role}')
            self.assertTrue(plots[1].origin.startswith(f'/imu/{role}/'))
        with tempfile.TemporaryDirectory() as tmp:
            blueprint.save('imu-four', str(Path(tmp)/'layout.rbl'))

    def test_other_modes_and_existing_replay_remain_image_only(self):
        for mode, enabled in [('aprilgrid', True), ('rgb', True), ('gesture', False)]:
            root = build_blueprint(self.cfg(['head', 'chest'], mode), imu=enabled).root_container
            self.assertEqual(len(root.contents), 2)
            self.assertTrue(all(len(r.contents) == 2 for r in root.contents))


if __name__ == '__main__':
    unittest.main()
