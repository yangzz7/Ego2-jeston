"""Exact GPU preprocessing; CPU fallback preserves frame content and ownership."""
import ctypes as C
import os
from pathlib import Path
import threading
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def backend():
    value=os.environ.get('EGO2_GESTURE_PREPROCESS_BACKEND','auto').lower()
    if value not in ('auto','cpu','cuda'):
        raise ValueError('EGO2_GESTURE_PREPROCESS_BACKEND must be auto, cpu or cuda')
    return 'cuda' if value=='auto' and Path('/etc/nv_tegra_release').is_file() else 'cpu' if value=='auto' else value


class CudaOperation:
    def __init__(self, shape, maps=None):
        self.lock=threading.RLock();self.handle=None
        self.shape=tuple(shape)
        if len(self.shape)!=3 or self.shape[2]!=3:raise ValueError('Expected HxWx3 BGR')
        h,w,_=self.shape
        xy,fraction=(None,None) if maps is None else cv2.convertMaps(*maps,cv2.CV_16SC2)
        oh,ow=(h,w) if xy is None else xy.shape[:2]
        self.output_shape=(oh,ow,3)
        self.lib=C.CDLL(os.environ.get('EGO2_GESTURE_PREPROCESS_LIBRARY',str(ROOT/'build/gesture_preprocess/libego2_gesture_preprocess.so')))
        self.lib.ego2_pre_create.argtypes=[C.c_int]*4+[C.c_void_p]*3+[C.c_size_t]
        self.lib.ego2_pre_create.restype=C.c_void_p
        self.lib.ego2_pre_run.argtypes=[C.c_void_p]*5+[C.c_size_t]
        self.lib.ego2_pre_run.restype=C.c_int
        self.lib.ego2_pre_sample.argtypes=[C.c_void_p]*4+[C.c_size_t]
        self.lib.ego2_pre_sample.restype=C.c_int
        self.lib.ego2_pre_destroy.argtypes=[C.c_void_p];self.lib.ego2_pre_destroy.restype=None
        error=C.create_string_buffer(1024)
        self.handle=self.lib.ego2_pre_create(w,h,ow,oh,None if xy is None else xy.ctypes.data,
            None if fraction is None else fraction.ctypes.data,error,len(error))
        if not self.handle:raise RuntimeError('CUDA preprocess init: '+error.value.decode())

    def run(self,image,lut=None):
        if image.dtype!=np.uint8 or image.shape!=self.shape:raise ValueError('Preprocessor input changed shape/dtype')
        image=np.ascontiguousarray(image)
        if lut is not None:
            lut=np.ascontiguousarray(lut,dtype=np.uint8)
            if lut.shape!=(3,256):raise ValueError('Expected 3x256 LUT')
        output=np.empty(self.output_shape,np.uint8)
        with self.lock:
            if not self.handle:raise RuntimeError('Preprocessor closed')
            error=C.create_string_buffer(1024)
            status=self.lib.ego2_pre_run(self.handle,image.ctypes.data,output.ctypes.data,
                                       None if lut is None else lut.ctypes.data,error,len(error))
            if status:raise RuntimeError('CUDA preprocess: '+error.value.decode())
        return output

    def sample(self,image):
        if image.dtype!=np.uint8 or image.shape!=(1024,1280,3) or image.shape!=self.shape:
            raise ValueError('GPU area sample requires 1024x1280x3 uint8')
        image=np.ascontiguousarray(image);result=np.empty((120,160,3),np.uint8)
        with self.lock:
            if not self.handle:raise RuntimeError('Preprocessor closed')
            error=C.create_string_buffer(1024)
            status=self.lib.ego2_pre_sample(self.handle,image.ctypes.data,result.ctypes.data,error,len(error))
            if status:raise RuntimeError('CUDA area sample: '+error.value.decode())
        return result

    def close(self):
        with self.lock:
            if self.handle:self.lib.ego2_pre_destroy(self.handle);self.handle=None

    def __del__(self):
        try:self.close()
        except Exception:pass


class Accelerator:
    def __init__(self,name):
        self.name=name;self.mode=backend();self.operation=None;self.failed=False;self.announced=False

    def apply(self,image,*,maps=None,lut=None,sample=False):
        if self.mode=='cpu' or self.failed:return None
        try:
            if self.operation is None or self.operation.shape!=image.shape:
                self.close();self.operation=CudaOperation(image.shape,maps)
            value=self.operation.sample(image) if sample else self.operation.run(image,lut)
            if not self.announced:
                print(f'[gesture-preprocess] {self.name}: CUDA; original LUT/map/rounding',flush=True)
                self.announced=True
            return value
        except Exception as exc:
            self.close();self.failed=True
            print(f'[gesture-preprocess] {self.name}: CPU fallback: {type(exc).__name__}: {exc}',flush=True)
            return None

    def close(self):
        if self.operation:self.operation.close();self.operation=None


class WhiteBalance:
    def __init__(self):self.accelerator=Accelerator('white balance')
    def __call__(self,image):
        # Retain the exact original area sampling, float64 gain and uint8 LUT.
        sample=self.accelerator.apply(image,sample=True) if image.shape==(1024,1280,3) else None
        if sample is None:
            sample=cv2.resize(image,(160,120),interpolation=cv2.INTER_AREA)
        means=sample.mean(axis=(0,1));gains=np.clip(means.mean()/np.maximum(means,1),.5,2)
        lut=np.stack([np.clip(np.arange(256)*g,0,255).astype(np.uint8) for g in gains])
        result=self.accelerator.apply(image,lut=lut)
        return result if result is not None else cv2.merge([cv2.LUT(image[:,:,c],lut[c]) for c in range(3)])
    def close(self):self.accelerator.close()
