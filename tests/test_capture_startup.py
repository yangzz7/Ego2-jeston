"""Only empty startup metadata may be skipped; valid/runtime frames stay exact."""
import threading
import unittest
from unittest.mock import MagicMock
import numpy as np
from ego2.common.sources import Source
from tests.test_capture_cuda import raw_frame

class StartupTests(unittest.TestCase):
    def source(self,frames,buffers=2):
        source=Source({'_base':'.','input':{'type':'cyper','preserve_raw':True,
            'preserve_display_raw':True}}, {},decode_images=False)
        source.capture_buffer_count=buffers
        source.cap=MagicMock();source.cap.read.side_effect=[(True,f) for f in frames]
        return source

    def test_only_empty_initial_frames_skipped_before_valid_sensor_frame(self):
        raw=raw_frame();bad=np.zeros_like(raw)
        s=self.source([bad,bad,raw,raw])
        pair=s.read(threading.Event())
        self.assertEqual(pair.sequence,1)
        self.assertEqual(s.cap.read.call_count,3)
        np.testing.assert_array_equal(pair.raw_transport,raw)
        self.assertEqual(len(pair.metadata['imu_samples']),7)
        self.assertEqual(s.read(threading.Event()).sequence,2)

    def test_runtime_empty_metadata_and_unknown_firmware_are_errors(self):
        raw=raw_frame();s=self.source([raw,np.zeros_like(raw)])
        s.read(threading.Event())
        with self.assertRaisesRegex(ValueError,'Unsupported/invalid'):s.read(threading.Event())
        s=self.source([raw_frame(fw=8)])
        with self.assertRaisesRegex(ValueError,'Unsupported/invalid'):s.read(threading.Event())
        self.assertEqual(s.cap.read.call_count,1)

    def test_startup_retry_is_bounded_and_single_buffer_policy_unchanged(self):
        raw=np.zeros_like(raw_frame())
        s=self.source([raw]*4)
        with self.assertRaisesRegex(RuntimeError,'only empty startup'):s.read(threading.Event())
        self.assertEqual(s.cap.read.call_count,4)
        s=self.source([raw],buffers=1)
        with self.assertRaises(ValueError):s.read(threading.Event())
        self.assertEqual(s.cap.read.call_count,1)

    def test_stop_and_driver_failure_do_not_substitute_images(self):
        s=self.source([np.zeros_like(raw_frame())]);stop=threading.Event();stop.set()
        self.assertIsNone(s.read(stop));self.assertEqual(s.count,0)
        s=self.source([]);s.cap.read.side_effect=None;s.cap.read.return_value=(False,None)
        with self.assertRaisesRegex(RuntimeError,'Camera read failed'):s.read(threading.Event())

if __name__=='__main__':unittest.main()
