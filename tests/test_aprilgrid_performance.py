"""Accuracy, same-frame recording and reacquisition guards for AprilGrid speedups."""
import copy
import pickle
import queue
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from ego2.aprilgrid.config import load_config
from ego2.aprilgrid.detector import Detector
from ego2.aprilgrid.temporal import Pipeline
from ego2.aprilgrid.tracking import Track
from ego2.aprilgrid.visualizer import projected_axes, projected_axes_batch
from ego2.common.sources import Pair
from ego2.multi_vio.mode_viewer import AprilGridSink, BatchRecording, UnifiedViewer, _encode_aprilgrid_original
from ego2.multi_vio.preview_transport import PreviewImage


class AprilGridPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.cfg, cls.cameras, cls.board = load_config()

    def test_fixed_maps_preserve_every_resampled_pixel_including_invalid_regions(self):
        d = Detector(self.cfg, self.cameras, self.board)
        self.addCleanup(d.close)
        rng = np.random.default_rng(423)
        for name, (_, views) in d.views.items():
            gray = rng.integers(0, 256, self.cameras[name].size[::-1], dtype=np.uint8)
            for view, fixed in zip(views, d.remap_tables[name]):
                options = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=128)
                np.testing.assert_array_equal(cv2.remap(gray, view[0], view[1], **options),
                                              cv2.remap(gray, *fixed, **options))

    def test_axes_batch_keeps_order_colors_and_clipping(self):
        poses = []
        for x, y, z in ((0, 0, .4), (.8, .1, .1), (-1, .3, .02), (0, 0, -.3)):
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(np.array([.1, -.4, .2]))[0]
            pose[:3, 3] = x, y, z
            poses.append(pose)
        for cam in self.cameras.values():
            old = [strip for pose in poses for strip in projected_axes(cam, pose, .08)]
            new = projected_axes_batch(cam, poses, .08)
            self.assertEqual(len(old), len(new))
            for (a, ca), (b, cb) in zip(old, new):
                self.assertEqual(ca, cb)
                np.testing.assert_array_equal(a, b)

    def test_empty_search_is_bounded_and_never_reuses_stale_measurements(self):
        p = Pipeline(copy.deepcopy(self.cfg), self.cameras, self.board, throttle_empty=True)
        self.addCleanup(p.close)
        images = tuple(np.full(cam.size[::-1]+(3,), 160, np.uint8) for cam in self.cameras.values())
        with patch.object(p.detector, 'observe', return_value=[]) as detect:
            for i in range(8):
                self.assertEqual(p.process(images, i/30), [])
                self.assertIsNone(p.track.measurement)
                self.assertIsNone(p.track.display_pose)
            self.assertEqual(detect.call_count, 1)
            p.process(images, .27)
            self.assertEqual(detect.call_count, 2)
            p.process(images, 0.)
            self.assertEqual(detect.call_count, 3)  # A reset must trigger a new search.

    def test_reliable_batch_preserves_original_recording_pixels_and_frame_times(self):
        q = queue.Queue(maxsize=2)
        rec = BatchRecording(q, 'chest', 0., demo=True, eye_names=('cam0','cam1'), reliable=True)
        sink = AprilGridSink(self.cfg, self.cameras, self.board, None, None, recording=rec)
        rng = np.random.default_rng(7)
        images = tuple(rng.integers(0,256,(24,32,3),dtype=np.uint8) for _ in range(2))
        expected = [image.copy() for image in images]
        track = Track(self.cfg['tracking'])
        with patch('rerun.Image.compress', side_effect=AssertionError('worker JPEG')):
            sink.log(Pair(images,.5,.5,17), .5, [], track, 1.)
        batch = pickle.loads(q.get_nowait())
        images[0][:] = 0
        entries = [row for row in batch['logs'] if row[0].endswith('/image')]
        self.assertEqual(len(entries), 2)
        for row, original in zip(entries, expected):
            value = row[1][0]
            self.assertIsInstance(value, PreviewImage)
            self.assertTrue(value.rerun_compress)
            np.testing.assert_array_equal(value.pixels, original)
            self.assertEqual(row[3]['chest_frame'], {'sequence':17})
            old = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            import rerun as rr
            old = rr.Image(old).compress(jpeg_quality=value.quality)
            new = _encode_aprilgrid_original(value.pixels,value.quality)
            self.assertTrue(old.blob.as_arrow_array().equals(new.blob.as_arrow_array()))
        viewer = UnifiedViewer.__new__(UnifiedViewer)
        viewer.cfg = {'viewer':{'jpeg_quality':85}}
        viewer.recording = MagicMock()
        enhanced = tuple(np.full_like(im, 55) for im in expected)
        with patch('ego2.multi_vio.mode_viewer._encode_aprilgrid_original', side_effect=AssertionError('redundant encoding')), \
             patch('ego2.multi_vio.mode_viewer.encode_image', return_value='encoded') as encoder:
            viewer._log_batch(batch, enhanced)
        self.assertEqual(encoder.call_count, 2)


if __name__ == '__main__': unittest.main()
