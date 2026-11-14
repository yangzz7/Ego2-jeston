"""Parallel eye encoding must retain complete, same-frame preview batches."""
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ego2.common.viewer import TIMELINE
from ego2.multi_vio.mode_viewer import UnifiedViewer
from ego2.multi_vio.preview_transport import PreviewImage


class PreviewEncodingTests(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(2)
        self.addCleanup(self.pool.shutdown, wait=True)
        self.viewer = UnifiedViewer.__new__(UnifiedViewer)
        self.viewer.encoder_pool = self.pool
        self.viewer.cfg = {'viewer': {'jpeg_quality': 90}}
        self.viewer.recording = MagicMock()
        self.images = [np.full((12, 16, 3), v, np.uint8) for v in (11, 22)]
        self.times = {TIMELINE: {'duration': 2.5}, 'head_frame': {'sequence': 73}}
        self.batch = {'role': 'head', 'received': 12.5, 'display_eyes': ('cam1', 'cam0'), 'logs': [
            ('/cameras/head/cam1/rgb', (PreviewImage(self.images[0], 90),), {}, self.times),
            ('/cameras/head/cam1/hand', ('skeleton-frame-73',), {}, self.times),
            ('/cameras/head/cam0/rgb', (PreviewImage(self.images[1], 90),), {}, self.times),
        ]}

    def test_second_eye_finishes_first_but_whole_batch_keeps_order_and_times(self):
        second_finished = threading.Event()
        def encode(image, quality):
            self.assertEqual(quality, 90)
            if image is self.images[0]:
                self.assertTrue(second_finished.wait(3), 'Eyes were encoded serially')
            else:
                self.viewer.recording.log.assert_not_called()
                second_finished.set()
            return 'jpeg-' + str(image[0, 0, 0])
        with patch('ego2.multi_vio.mode_viewer.encode_image', side_effect=encode):
            self.viewer._log_batch(self.batch)
        calls = self.viewer.recording.log.call_args_list
        self.assertEqual([c.args[0] for c in calls], [x[0] for x in self.batch['logs']])
        self.assertEqual([c.args[1] for c in calls], ['jpeg-11', 'skeleton-frame-73', 'jpeg-22'])
        self.assertEqual([c.kwargs for c in self.viewer.recording.set_time.call_args_list],
                         [{'duration': 2.5}, {'sequence': 73}] * 3)
        for image, value in zip(self.images, (11, 22)):
            np.testing.assert_array_equal(image, value)

    def test_encoding_failure_publishes_no_partial_frame(self):
        def encode(image, quality):
            if image is self.images[1]:
                raise RuntimeError('jpeg failed')
            return 'jpeg-11'
        with patch('ego2.multi_vio.mode_viewer.encode_image', side_effect=encode):
            with self.assertRaisesRegex(RuntimeError, 'jpeg failed'):
                self.viewer._log_batch(self.batch)
        self.viewer.recording.log.assert_not_called()
        self.viewer.recording.set_time.assert_not_called()

    def test_isp_replacements_keep_frame_and_recording_cutoff(self):
        enhanced = [np.full_like(image, value) for image, value in zip(self.images, (33, 44))]
        with patch('ego2.multi_vio.mode_viewer.encode_image',
                   side_effect=lambda image, quality: int(image[0, 0, 0])):
            self.viewer._log_batch(self.batch, enhanced, cutoff=12.)
        self.assertEqual([c.args[1] for c in self.viewer.recording.log.call_args_list],
                         [33, 'skeleton-frame-73', 44])
        self.assertEqual([c.kwargs for c in self.viewer.recording.set_time.call_args_list],
                         [{'duration': .5}, {'sequence': 73}] * 3)
        self.assertEqual(self.batch['logs'][1][1], ('skeleton-frame-73',))
        self.assertEqual(self.times[TIMELINE], {'duration': 2.5})


if __name__ == '__main__':
    unittest.main()
