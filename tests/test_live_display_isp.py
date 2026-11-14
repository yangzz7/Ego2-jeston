"""Live-only enhancement: immutable algorithm/archive data and frame pairing."""
import copy
import json
from pathlib import Path
import pickle
import queue
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import rerun as rr

from ego2.common.display_isp import DisplayIsp, create_display_isp, settings_for
from ego2.common.display_raw import DisplayRawReader
from ego2.common.paths import ROOT, BUILD
from ego2.common.sources import Pair, Source, decode_cyper, white_balance
from ego2.multi_vio.config import MULTI_CONFIG
from ego2.multi_vio.modes import load_mode_config
from ego2.multi_vio.mode_viewer import BatchRecording, GestureSink, UnifiedViewer


def raw_frame(sequence=6, size=(1024,1280)):
    raw = np.full((*size,2),70,np.uint8)
    raw[:,:,1] = 120
    raw[-1] = 0
    raw[-1,0,0],raw[-1,1,0],raw[-1,4,0] = 2,sequence,10
    return raw


class FakeIsp:
    def __init__(self, block=False, fail=False):
        self.entered, self.release = threading.Event(), threading.Event()
        self.closed, self.fail = False, fail
        if not block: self.release.set()

    def process_transport(self, raw):
        self.entered.set()
        if not self.release.wait(3): raise TimeoutError('Test did not release ISP')
        if self.fail: raise ValueError('bad metadata')
        return tuple(np.repeat(raw[:,:,i:i+1],3,axis=2) for i in range(2))

    def close(self): self.closed = True


class QueueTests(unittest.TestCase):
    def test_slow_worker_owns_raw_and_drops_whole_pending_frame(self):
        isp = FakeIsp(block=True)
        worker = DisplayIsp(isp,max_fps=1000)
        original = tuple(np.full((8,8,3),i,np.uint8) for i in (1,2))
        shown = []
        try:
            for number in (1,2,3):
                raw = np.full((8,8,2),number,np.uint8)
                worker.submit(raw,original,lambda images,n=number:shown.append((n,images)))
                if number==1: self.assertTrue(isp.entered.wait(2))
                raw[:] = 99
            self.assertFalse(shown)
        finally:
            isp.release.set()
            worker.close()
        self.assertEqual([n for n,_ in shown],[1,3])
        for n,images in shown: self.assertTrue(np.all(images[0]==n))
        self.assertEqual(worker.dropped,1)
        self.assertTrue(isp.closed)
        self.assertTrue(np.all(original[0]==1))

    def test_fallback_is_current_frame_and_reconstruction_order_is_explicit(self):
        for missing,fail in ((True,False),(False,True),(False,False)):
            with self.subTest(missing=missing,fail=fail):
                worker=DisplayIsp(FakeIsp(fail=fail))
                original=(np.zeros((8,8,3),np.uint8),)*2
                raw=np.full((8,8,2),31,np.uint8);raw[:,:,1]=97
                outputs=[]
                worker.submit(None if missing else raw,original,outputs.append,order=(1,0))
                worker.close()
                if missing or fail: self.assertIs(outputs[0],original)
                else:
                    self.assertTrue(np.all(outputs[0][0]==97))
                    self.assertTrue(np.all(outputs[0][1]==31))

    def test_vio_snapshots_overlays_but_keeps_algorithm_arrays_untouched(self):
        from ego2.multi_vio.visualizer import Visualizer
        viewer=Visualizer.__new__(Visualizer)
        isp=FakeIsp(block=True)
        worker=DisplayIsp(isp)
        viewer.display_processors={'head':worker}
        viewer._images=MagicMock()
        packet={'frame_id':5,'capture_time':.5}
        observations=[SimpleNamespace(pixels=np.array([[1.,2.]]))]
        track=SimpleNamespace(display_pose=np.eye(4))
        images=tuple(np.full((8,8,3),i,np.uint8) for i in (10,20))
        try:
            viewer.images('head',packet,images,observations,track,'tracking',4,
                          raw_transport=np.full((8,8,2),70,np.uint8))
            self.assertTrue(isp.entered.wait(2))
            observations[0].pixels[:]=99; track.display_pose[:]=99;packet['frame_id']=99
        finally:
            isp.release.set();worker.close()
        args=viewer._images.call_args.args
        self.assertEqual(args[1]['frame_id'],5)
        np.testing.assert_array_equal(args[3][0].pixels,[[1.,2.]])
        np.testing.assert_array_equal(args[4].display_pose,np.eye(4))
        self.assertTrue(np.all(images[0]==10))


class TransportTests(unittest.TestCase):
    def test_display_raw_does_not_enable_raw_recording_or_change_decode(self):
        raw=raw_frame(size=(8,160));original=raw.copy()
        profile={'_base':'.','input':{'type':'cyper','width':160,'height':8,'fps':30,
                                    'preserve_raw':False,'preserve_display_raw':True}}
        source=Source(profile,{})
        source.cap=MagicMock();source.cap.read.return_value=(True,raw)
        with patch('ego2.common.sources.time.monotonic',return_value=12.):
            pair=source.read(threading.Event())
        self.assertIsNone(pair.raw_transport)
        np.testing.assert_array_equal(pair.display_transport,original)
        expected,_=decode_cyper(original,160,8)
        for a,b in zip(pair.images,expected): np.testing.assert_array_equal(a,white_balance(b))
        np.testing.assert_array_equal(raw,original)
        raw[:]=0
        np.testing.assert_array_equal(pair.display_transport,original)

    def test_native_ring_preserves_metadata_order_and_rejects_overwritten_slot(self):
        executable=BUILD/'bin/test_display_raw'
        self.assertTrue(executable.exists(),'Build the native display RAW test first')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ring'
            subprocess.run([str(executable),str(path)],check=True)
            reader=DisplayRawReader(path)
            try:
                frame=reader.read(4)
                self.assertEqual(frame[100,100].tolist(),[31,97])
                self.assertEqual(frame[-1,0].tolist(),[2,97])
                self.assertEqual(frame[-1,-1].tolist(),[123,97])
                self.assertIsNone(reader.read(1))
                self.assertIsNone(reader.read(5))
                with path.open('r+b') as writer:
                    import fcntl
                    fcntl.flock(writer,fcntl.LOCK_EX)
                    self.assertIsNone(reader.read(4))
                    fcntl.flock(writer,fcntl.LOCK_UN)
            finally: reader.close()

    def test_live_batch_enhances_both_eyes_without_changing_archived_batch(self):
        q=queue.Queue()
        bridge=BatchRecording(q,'head',0.,demo=True,eye_names=('cam0','cam1'))
        sink=GestureSink({'viewer':{'jpeg_quality':90}},bridge)
        raw=raw_frame()
        originals=tuple(np.full((1024,1280,3),v,np.uint8) for v in (10,20))
        pair=Pair(originals,.5,.5,15,display_transport=raw)
        hand=SimpleNamespace(id=1,handedness='Right',gesture='Palm',score=.9,pinch='released',
                             points=np.full((21,2),100.),visible=np.ones(21,dtype=bool))
        sink.log(pair,.5,([hand],[]))
        blob=q.get_nowait();before=bytes(blob)
        archive=UnifiedViewer.__new__(UnifiedViewer);archive.recording=MagicMock()
        viewer=UnifiedViewer.__new__(UnifiedViewer);viewer.recording=MagicMock()
        viewer.cfg={'viewer':{'jpeg_quality':90}}
        viewer.file_viewer=archive
        worker=DisplayIsp(FakeIsp())
        viewer.display_processors={'head':worker}
        viewer.consume(blob);worker.close()
        self.assertEqual(blob,before)
        old={c.args[0]:c.args[1] for c in archive.recording.log.call_args_list}
        new={c.args[0]:c.args[1] for c in viewer.recording.log.call_args_list}
        for cam,value,base in [('cam0',70,10),('cam1',120,20)]:
            path=f'/cameras/head/{cam}/rgb'
            def pixel(item):
                data=bytes(item.blob.as_arrow_array().to_pylist()[0])
                return cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)[100,100]
            np.testing.assert_allclose(pixel(new[path]),value,atol=1)
            np.testing.assert_allclose(pixel(old[path]),base,atol=1)
        path='/cameras/head/cam0/hands/1/points'
        self.assertTrue(new[path].positions.as_arrow_array().equals(old[path].positions.as_arrow_array()))
        self.assertEqual(viewer.recording.set_time.call_args_list,archive.recording.set_time.call_args_list)
        self.assertTrue(np.all(pair.images[0]==10))

    def test_reconstruction_live_and_file_images_are_separate(self):
        from ego2.reconstruction.visualizer import Visualizer
        viewer=Visualizer.__new__(Visualizer)
        viewer._rgb=MagicMock();viewer.file_viewer=MagicMock()
        viewer.display_processor=DisplayIsp(FakeIsp())
        originals=(np.zeros((8,8,3),np.uint8),)*2
        raw=np.full((8,8,2),31,np.uint8);raw[:,:,1]=97
        viewer.rgb(originals,1.25,raw_transport=raw)
        viewer.display_processor.close()
        self.assertIs(viewer.file_viewer.rgb.call_args.args[0],originals)
        images,stamp=viewer._rgb.call_args.args
        self.assertEqual(stamp,1.25)
        self.assertTrue(np.all(images[0]==97))
        self.assertTrue(np.all(images[1]==31))

    def test_configured_eye_binding_is_preserved_in_enhanced_batch(self):
        q=queue.Queue()
        bridge=BatchRecording(q,'head',0.,demo=True,eye_names=('cam1','cam0'))
        pair=Pair((np.zeros((8,8,3),np.uint8),)*2,0.,0.,0,
                  display_transport=np.zeros((8,8,2),np.uint8))
        GestureSink({'viewer':{'jpeg_quality':90}},bridge).log(pair,0.,([],[]))
        batch=pickle.loads(q.get_nowait())
        self.assertEqual(batch['display_eyes'],('cam1','cam0'))
        viewer=UnifiedViewer.__new__(UnifiedViewer);viewer.recording=MagicMock()
        viewer.cfg={'viewer':{'jpeg_quality':90}}
        viewer._log_batch(batch,tuple(np.full((8,8,3),v,np.uint8) for v in (31,97)))
        frames={c.args[0]:c.args[1] for c in viewer.recording.log.call_args_list}
        for cam,value in [('cam1',31),('cam0',97)]:
            encoded=bytes(frames[f'/cameras/head/{cam}/rgb'].blob.as_arrow_array().to_pylist()[0])
            image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
            np.testing.assert_allclose(image,value,atol=1)


class ConfigurationAndNativeIspTests(unittest.TestCase):
    def test_all_sensia_modes_use_same_display_policy_and_missing_profile_falls_back(self):
        for mode in ('vio','gesture','aprilgrid','reconstruction'):
            cfg,devices=load_mode_config(MULTI_CONFIG,mode=mode,selection={'head':'s200032'})
            settings=settings_for(cfg,'s200032')
            self.assertTrue(settings['enabled'])
            self.assertTrue(Path(settings['library']).is_file())
            if mode=='reconstruction':
                self.assertEqual(settings_for(devices['head']['profile'],'s200032'),settings)
            self.assertIsNone(create_display_isp(cfg,'s200032',SimpleNamespace(demo=True)))
            self.assertIsNone(create_display_isp(cfg,'s200032',SimpleNamespace(headless=True)))
        self.assertIsNone(create_display_isp(cfg,'s299999',SimpleNamespace()))

    def test_real_isp_firmware_3_to_7_and_input_immutability(self):
        from cyperstereo_isp_portable.python.cyperstereo_isp import Isp
        cfg,_=load_mode_config(MULTI_CONFIG,mode='gesture',selection={'head':'s200032'})
        settings=settings_for(cfg,'s200032')
        with Isp(config=settings['config'],library=settings['library']) as isp:
            outputs={}
            for version in range(3,8):
                raw=raw_frame(version)
                raw[0:-1:2,0::2,0]=150;raw[1:-1:2,1::2,0]=30
                before=raw.copy()
                algorithm,_=decode_cyper(raw,1280,1024)
                saved=copy.deepcopy(algorithm)
                isp.reset()
                outputs[version]=isp.process_transport(raw)
                np.testing.assert_array_equal(raw,before)
                for a,b,display in zip(algorithm,saved,outputs[version]):
                    np.testing.assert_array_equal(a,b)
                    self.assertEqual(display.shape,a.shape)
                    self.assertGreater(np.abs(display.astype(float)-a).mean(),5)
            for v6,v7 in zip(outputs[6],outputs[7]): np.testing.assert_array_equal(v6,v7)
            raw=raw_frame(8)
            with self.assertRaisesRegex(RuntimeError,'unsupported'): isp.process_transport(raw)


if __name__=='__main__': unittest.main()
