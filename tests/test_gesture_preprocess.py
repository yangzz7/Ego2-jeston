"""Fallback must preserve the exact original pixels, maps and input ownership."""
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ego2.common.sources import white_balance
from ego2.gesture.preprocess import Accelerator, WhiteBalance, backend
from ego2.gesture.rectification import Rectifier
from tests.test_gesture import test_camera


class PreprocessTests(unittest.TestCase):
    def test_cpu_mode_does_not_load_cuda_and_preserves_pixels(self):
        with patch.dict(os.environ, EGO2_GESTURE_PREPROCESS_BACKEND='cpu'), \
             patch('ego2.gesture.preprocess.CudaOperation', side_effect=AssertionError('CUDA loaded')):
            balance = WhiteBalance()
            for shape in [(27, 38, 3), (1024, 1280, 3)]:
                image = np.random.default_rng(2).integers(0, 256, shape, dtype=np.uint8)
                original = image.copy()
                np.testing.assert_array_equal(balance(image), white_balance(image))
                np.testing.assert_array_equal(image, original)
            balance.close()

    def test_gpu_failure_falls_back_for_same_frame_only_once(self):
        with patch.dict(os.environ, EGO2_GESTURE_PREPROCESS_BACKEND='cuda'), \
             patch('ego2.gesture.preprocess.CudaOperation', side_effect=RuntimeError('no device')) as create:
            balance = WhiteBalance()
            for value in (4, 87, 211):
                image = np.full((1024, 1280, 3), value, np.uint8)
                np.testing.assert_array_equal(balance(image), white_balance(image))
            self.assertTrue(balance.accelerator.failed)
            self.assertEqual(create.call_count, 1)
            balance.close()

    def test_rectifier_failure_uses_original_map_and_color_order(self):
        with patch.dict(os.environ, EGO2_GESTURE_PREPROCESS_BACKEND='cuda'), \
             patch('ego2.gesture.preprocess.CudaOperation', side_effect=RuntimeError('no runtime')):
            rectifier = Rectifier(test_camera(), {'camera': 'cam0'},
                                  dict(width=320, height=240, hfov=75, pitch=0))
            image = np.random.default_rng(5).integers(0, 256, (480, 640, 3), dtype=np.uint8)
            original = image.copy()
            result = rectifier.prepare(image)
            expected = cv2.cvtColor(cv2.remap(image, *rectifier.maps.xy[0], cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(image, original)
            rectifier.close()

    def test_map_rebuild_releases_previous_device_maps(self):
        with patch.dict(os.environ, EGO2_GESTURE_PREPROCESS_BACKEND='cpu'):
            rectifier = Rectifier(test_camera(), {'camera': 'cam0'},
                                  dict(width=320, height=240, hfov=75, pitch=0))
        with patch.object(rectifier.accelerator, 'close') as close:
            rectifier.prepare(np.zeros((480, 640, 3), np.uint8))
            rectifier.prepare(np.zeros((480, 640, 3), np.uint8))
            rectifier.prepare(np.zeros((240, 320, 3), np.uint8))
            self.assertEqual(close.call_count, 2)

    def test_runtime_failure_releases_context(self):
        with patch.dict(os.environ, EGO2_GESTURE_PREPROCESS_BACKEND='cuda'), \
             patch('ego2.gesture.preprocess.CudaOperation') as create:
            operation = create.return_value
            operation.run.side_effect = RuntimeError('kernel failed')
            accelerator = Accelerator('test')
            self.assertIsNone(accelerator.apply(np.zeros((20, 20, 3), np.uint8)))
            operation.close.assert_called_once()
            self.assertTrue(accelerator.failed)
            self.assertIsNone(accelerator.operation)

    def test_invalid_backend_is_explicit(self):
        with patch.dict(os.environ, EGO2_GESTURE_PREPROCESS_BACKEND='bad'):
            with self.assertRaises(ValueError):
                backend()


if __name__ == '__main__':
    unittest.main()
