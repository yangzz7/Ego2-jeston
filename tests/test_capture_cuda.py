"""Opt-in CUDA checks: EGO2_TEST_CAPTURE_CUDA=1 python -m unittest tests.test_capture_cuda."""
import concurrent.futures
import os
import threading
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from ego2.common.cyper_cuda import CudaDecoder, capture_backend
from ego2.common.sources import Source, decode_cyper, white_balance
from ego2.common.cyper_metadata import CyperMetadataDecoder


def raw_frame(w=1280,h=1024,fw=7,seed=7):
    raw=np.random.default_rng(seed).integers(0,256,(h,w,2),dtype=np.uint8)
    raw[-1]=0
    words=raw[-1].reshape(-1).view('<u2')
    words[:2]=[2,fw]
    if w>=135:
        words[4]=12
        for i in range(13 if fw==5 else 7):
            words[5+9*i:14+9*i]=[1000+i*50,12,100,65500,5000,10,65530,30,16]
    return raw


@unittest.skipUnless(os.environ.get('EGO2_TEST_CAPTURE_CUDA')=='1','requires real CUDA device')
class CaptureCudaTests(unittest.TestCase):
    def test_pixel_exact_all_firmware_phases_sizes_and_input_unchanged(self):
        for w,h in [(2,3),(3,3),(4,5),(31,19),(160,8),(1280,1024)]:
            with CudaDecoder(w,h) as decoder:
                for fw in range(3,8):
                    raw=raw_frame(w,h,fw); before=raw.copy()
                    for phases in ['auto',['RG','BG'],['GR','GB']]:
                        expected,efw=decode_cyper(raw,w,h,phases)
                        actual,afw=decoder.decode(raw,phases)
                        self.assertEqual(efw,afw)
                        for a,b in zip(expected,actual): np.testing.assert_array_equal(a,b)
                    np.testing.assert_array_equal(raw,before)

    def test_extreme_pixels_border_metadata_and_white_balance(self):
        with CudaDecoder(1280,1024) as decoder:
            for value in [0,1,127,254,255]:
                raw=raw_frame();raw[:-1]=value
                raw[1,1]=[255,0];raw[-3,-2]=[1,254]
                expected,_=decode_cyper(raw,1280,1024);actual,_=decoder.decode(raw)
                for a,b in zip(expected,actual):
                    np.testing.assert_array_equal(a,b)
                    np.testing.assert_array_equal(white_balance(a),white_balance(b))

    def test_concurrent_independent_and_shared_contexts_own_outputs(self):
        raw=[raw_frame(seed=s) for s in range(8)]
        expected=[decode_cyper(r,1280,1024)[0] for r in raw]
        decoders=[CudaDecoder(1280,1024) for _ in range(4)]
        try:
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                for shared in (False,True):
                    actual=list(pool.map(lambda i:decoders[0 if shared else i%4].decode(raw[i])[0],range(8)))
                    for a,b in zip(expected,actual):
                        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
                    self.assertFalse(np.shares_memory(actual[0][0],actual[4][0]))
        finally:
            for d in decoders:d.close()

    def test_fused_stereo_white_balance_is_exact_and_does_not_change_raw(self):
        with CudaDecoder(1280,1024) as decoder:
            for seed in range(30):
                raw=raw_frame(fw=3+seed%5,seed=seed+100)
                if seed<5:raw[:-1]=[0,1,127,254,255][seed]
                if seed%3==0:raw[:-1,:,seed%2]//=128
                before=raw.copy()
                phases='auto' if seed%2 else ['GR','GB']
                expected,fw=decode_cyper(raw,1280,1024,phases)
                actual,got_fw=decoder.decode(raw,phases,white_balance=True)
                self.assertEqual(got_fw,fw)
                for a,b in zip(actual,expected):np.testing.assert_array_equal(a,white_balance(b))
                saved=[x.copy() for x in actual]
                again,_=decoder.decode(raw,phases)
                for a,b in zip(again,expected):np.testing.assert_array_equal(a,b)
                for a,b in zip(actual,saved):np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(raw,before)

    def test_bad_input_and_closed_context(self):
        with CudaDecoder(160,8) as decoder:
            for raw in [raw_frame(160,8).astype(np.uint16),raw_frame(159,8),raw_frame(160,8,8)]:
                with self.assertRaises(ValueError):decoder.decode(raw)
            with self.assertRaises(ValueError):decoder.decode(raw_frame(160,8),['bad','RG'])
        with self.assertRaises(RuntimeError):decoder.decode(raw_frame(160,8))
        decoder.close()
        with self.assertRaises(RuntimeError):CudaDecoder(1,8)

    def test_shared_source_uses_gpu_preserves_recording_raw_metadata_and_images(self):
        raw=raw_frame()
        profile={'input':{'type':'cyper','device':'/dev/fake','width':1280,'height':1024,
                         'preserve_raw':True,'preserve_display_raw':True},'_base':'.'}
        cap=MagicMock();cap.isOpened.return_value=True;cap.read.return_value=(True,raw)
        cap.get.side_effect=lambda key:0 if key==16 else 30
        source=Source(profile,{})
        source.fuse_white_balance=True
        with patch.dict(os.environ,{'EGO2_CAPTURE_BACKEND':'cuda'}),patch('cv2.VideoCapture',return_value=cap):
            try:
                source.open();self.assertIsInstance(source.cuda_decoder,CudaDecoder)
                pair=source.read(threading.Event())
                np.testing.assert_array_equal(pair.raw_transport,raw)
                np.testing.assert_array_equal(pair.display_transport,raw)
                expected,_=decode_cyper(raw,1280,1024)
                for x,y in zip(pair.images,expected):np.testing.assert_array_equal(x,white_balance(y))
                expected_metadata=CyperMetadataDecoder().decode(raw,1280,1024)
                for key,value in expected_metadata.items():self.assertEqual(pair.metadata[key],value)
            finally:source.close()


class FusedFallbackTests(unittest.TestCase):
    def test_failed_fused_stage_retries_same_frame_and_retains_metadata(self):
        raw=raw_frame();original=raw.copy()
        expected,firmware=decode_cyper(raw,1280,1024)
        source=Source({'_base':'.','input':{'type':'cyper','preserve_raw':True}}, {})
        source.cap=MagicMock();source.cap.read.return_value=(True,raw)
        source.cuda_decoder=MagicMock()
        source.cuda_decoder.decode.side_effect=[RuntimeError('fused unavailable'),(expected,firmware),(expected,firmware)]
        source.fuse_white_balance=True
        a=source.read(threading.Event());b=source.read(threading.Event())
        self.assertFalse(source.fuse_white_balance)
        self.assertEqual(source.cuda_decoder.decode.call_count,3)
        calls=source.cuda_decoder.decode.call_args_list
        self.assertEqual(calls[0].kwargs,{'white_balance':True})
        self.assertFalse(calls[1].kwargs)
        self.assertFalse(calls[2].kwargs)
        for call in calls:self.assertIs(call.args[0],raw)
        for pair in (a,b):
            for got,ref in zip(pair.images,expected):np.testing.assert_array_equal(got,white_balance(ref))
            np.testing.assert_array_equal(pair.raw_transport,original)
            self.assertEqual(pair.metadata['raw_metadata_words'],CyperMetadataDecoder().decode(raw,1280,1024)['raw_metadata_words'])
        np.testing.assert_array_equal(raw,original)


class BackendSelectionTests(unittest.TestCase):
    def test_explicit_cpu_and_invalid_selection(self):
        with patch.dict(os.environ,{'EGO2_CAPTURE_BACKEND':'cpu'}):self.assertEqual(capture_backend(),'cpu')
        with patch.dict(os.environ,{'EGO2_CAPTURE_BACKEND':'typo'}):
            with self.assertRaises(ValueError):capture_backend()
