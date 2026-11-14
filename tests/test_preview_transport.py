"""Verify disposable preview optimization cannot change frame/recording data."""
import multiprocessing as mp
import pickle
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from ego2.common.sources import Pair
from ego2.common.display_isp import DisplayIsp
from ego2.multi_vio.mode_viewer import BatchRecording, GestureSink, UnifiedViewer
from ego2.multi_vio.preview_transport import PreviewRing, pack_images, unpack_images, PreviewImage


def child_write(spec, pipe):
    ring = PreviewRing(spec)
    try:
        pipe.send(ring.write([np.full((12, 16, 2), 77, np.uint8)]))
    finally:
        ring.close()
        pipe.close()


class PreviewTransportTests(unittest.TestCase):
    def setUp(self):
        self.context = mp.get_context('spawn')
        self.ring = PreviewRing.create(self.context)
        self.addCleanup(self.ring.close)

    def test_snapshot_is_owned_and_stale_generation_rejected(self):
        original = np.full((8, 10, 2), 17, np.uint8)
        reference = self.ring.write([original])
        saved = self.ring.read(reference)
        original[:] = 88
        for n in range(3):
            self.ring.write([np.full_like(original, n)])
        self.assertIsNone(self.ring.read(reference))
        np.testing.assert_array_equal(saved[0], 17)
        self.assertTrue(saved[0].flags.owndata)

    def test_spawned_writer_shares_only_descriptors(self):
        reader, writer = self.context.Pipe(duplex=False)
        process = self.context.Process(target=child_write, args=(self.ring.spec, writer))
        process.start()
        self.assertTrue(reader.poll(10))
        reference = reader.recv()
        process.join(10)
        self.assertEqual(process.exitcode, 0)
        np.testing.assert_array_equal(self.ring.read(reference)[0], 77)
        reader.close(); writer.close()

    def test_busy_slot_is_dropped_without_blocking_or_partial_snapshot(self):
        reference = self.ring.write([np.zeros((8,8,2),np.uint8)])
        self.ring.lock.acquire()
        try:
            self.assertIsNone(self.ring.write([np.ones((8,8,2),np.uint8)]))
            self.assertIsNone(self.ring.read(reference))
        finally:
            self.ring.lock.release()
        self.assertIsNotNone(self.ring.read(reference))

    def test_reliable_recording_rejects_disposable_transport(self):
        with self.assertRaisesRegex(ValueError, 'durable'):
            BatchRecording(queue.Queue(), 'head', 0, reliable=True, transport=self.ring)

    def test_no_worker_jpeg_and_matching_original_fallback(self):
        q = queue.Queue()
        rec = BatchRecording(q,'head',0.,demo=True,eye_names=('cam1','cam0'),transport=self.ring)
        sink = GestureSink({'viewer':{'jpeg_quality':90}},rec)
        originals = tuple(np.full((1024,1280,3),v,np.uint8) for v in (10,20))
        raw = np.full((1024,1280,2),33,np.uint8)
        pair = Pair(originals,.5,.5,7,display_transport=raw)
        with patch('cv2.imencode', side_effect=AssertionError('Worker must not encode')):
            sink.log(pair,.5,([],[]))
        blob=q.get_nowait()
        self.assertLess(len(blob), 30_000)
        packed=pickle.loads(blob)
        self.assertNotIn('display_raw',packed)
        batch=unpack_images(packed,self.ring)
        original_serialized=pickle.dumps(batch)
        viewer=UnifiedViewer.__new__(UnifiedViewer)
        viewer.recording=MagicMock(); viewer.cfg={'viewer':{'jpeg_quality':90}}
        viewer.display_processors={};viewer.file_viewer=None
        with patch('ego2.multi_vio.mode_viewer.encode_image',return_value='encoded') as encode:
            viewer.consume(batch)
        self.assertEqual(encode.call_count,2)
        np.testing.assert_array_equal(encode.call_args_list[0].args[0],originals[0])
        np.testing.assert_array_equal(encode.call_args_list[1].args[0],originals[1])
        self.assertEqual(original_serialized,pickle.dumps(batch))
        np.testing.assert_array_equal(raw,33)
        for image,value in zip(pair.images,(10,20)): np.testing.assert_array_equal(image,value)
        self.assertEqual(batch['logs'][0][3]['head_frame'],{'sequence':7})

    def test_isp_replacement_encodes_only_enhanced_pair_at_original_times(self):
        q=queue.Queue()
        rec=BatchRecording(q,'head',0.,demo=True,eye_names=('cam1','cam0'),transport=self.ring)
        raw=np.full((1024,1280,2),33,np.uint8)
        originals=tuple(np.full((1024,1280,3),v,np.uint8) for v in (10,20))
        pair=Pair(originals,.5,.5,9,display_transport=raw)
        GestureSink({'viewer':{'jpeg_quality':90}},rec).log(pair,.5,([],[]))
        batch=unpack_images(pickle.loads(q.get_nowait()),self.ring)
        viewer=UnifiedViewer.__new__(UnifiedViewer);viewer.recording=MagicMock();viewer.cfg={'viewer':{'jpeg_quality':85}}
        enhanced=tuple(np.full((1024,1280,3),v,np.uint8) for v in (31,97))
        with patch('ego2.multi_vio.mode_viewer.encode_image',return_value='encoded') as encode:
            viewer._log_batch(batch,enhanced)
        self.assertEqual(encode.call_count,2)
        self.assertIs(encode.call_args_list[0].args[0],enhanced[0])
        self.assertIs(encode.call_args_list[1].args[0],enhanced[1])
        self.assertEqual(viewer.recording.log.call_args_list[0].args[0],'/cameras/head/cam1/rgb')
        self.assertIn(('head_frame',),[c.args for c in viewer.recording.set_time.call_args_list])


class BackendFallbackTests(unittest.TestCase):
    def test_missing_cuda_runtime_keeps_original_cpu_isp(self):
        from pathlib import Path
        import tempfile
        from ego2.common.display_isp import create_display_isp
        cpu=MagicMock();cpu.backend='cpu'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            library=root/'build/cyperstereo_isp_display/libcyperstereo_isp_c.so'
            library.parent.mkdir(parents=True);library.touch()
            with patch('ego2.common.display_isp.ROOT',root), patch.dict('os.environ',{'EGO2_DISPLAY_BACKEND':'auto'}), \
                 patch('cyperstereo_isp_portable.python.cyperstereo_isp.Isp',side_effect=[OSError('missing CUDA runtime'),OSError('missing CUDA runtime'),cpu]) as factory:
                worker=create_display_isp({'viewer':{'display_isp':{'enabled':True}}},'s200032',SimpleNamespace())
                self.assertIsNotNone(worker)
                worker.close()
                self.assertEqual(factory.call_count,3)
                self.assertNotIn('backend',factory.call_args.kwargs)
                self.assertEqual(factory.call_args.kwargs['library'],str(root/'build/cyperstereo_isp/libcyperstereo_isp_c.so'))
                cpu.close.assert_called_once()


if __name__=='__main__': unittest.main()
