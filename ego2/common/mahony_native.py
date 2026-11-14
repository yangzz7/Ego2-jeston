"""Opt-in native CPU batch Mahony; reference Python implementation is retained."""
import ctypes as ct
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from .mahony import Mahony,Settings,from_rotation

SOURCE=Path(__file__).with_name('native')/'mahony.cpp'
KEY=hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
LIBRARY=Path(__file__).resolve().parents[2]/'build'/'mahony'/KEY/'libego2_mahony.so'
DOUBLE=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')

def build():
    LIBRARY.parent.mkdir(parents=True,exist_ok=True)
    fd,path=tempfile.mkstemp(prefix='libmahony-',suffix='.so',dir=LIBRARY.parent);os.close(fd)
    try:
        subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared','-Wall','-Wextra',str(SOURCE),'-o',path],check=True)
        os.replace(path,LIBRARY)
    finally:Path(path).unlink(missing_ok=True)
    print(LIBRARY)
    return LIBRARY

def load():
    lib=ct.CDLL(str(LIBRARY))
    lib.ego2_mahony_abi.restype=ct.c_int
    if lib.ego2_mahony_abi()!=1:raise RuntimeError('Unexpected Mahony native ABI')
    lib.ego2_mahony_new.argtypes=[DOUBLE,DOUBLE];lib.ego2_mahony_new.restype=ct.c_void_p
    lib.ego2_mahony_delete.argtypes=[ct.c_void_p];lib.ego2_mahony_delete.restype=None
    lib.ego2_mahony_batch.argtypes=[ct.c_void_p,DOUBLE,ct.c_size_t,DOUBLE,DOUBLE];lib.ego2_mahony_batch.restype=ct.c_int
    lib.ego2_mahony_reset.argtypes=[ct.c_void_p,ct.c_int,ct.c_int];lib.ego2_mahony_reset.restype=None
    lib.ego2_mahony_info.argtypes=[ct.c_void_p,DOUBLE];lib.ego2_mahony_info.restype=None
    lib.ego2_mahony_state.argtypes=[ct.c_void_p,DOUBLE];lib.ego2_mahony_state.restype=ct.c_int
    return lib

def status_text(info):
    code,segment=map(int,info[:2])
    if code==0:return 'WAITING: hold camera still for 2.5 s'
    if code==2:return 'WAITING: IMU gap; hold still to rebuild reference'
    if code==6:return 'WAITING: attitude backlog; hold still to rebuild reference'
    if code==3:return 'WAITING: data interrupted; hold still to rebuild reference'
    quality=f'ESTIMATED | IMU gaps <= {info[4]*1000:.0f} ms' if code==4 else 'VALID | cam0 relative'
    return quality+f' | reference #{segment}'+(' (rebuilt)' if segment>1 else '')

class NativeMahony:
    backend='native-cpu'
    def __init__(self,camera_from_imu,settings=None):
        self.handle=None
        self.settings=settings or Settings()
        # One-time validation shares the reference's conventions and bounds.
        validated=Mahony(camera_from_imu,self.settings)
        self.lib=load();self.handle=None
        self.handle=self.lib.ego2_mahony_new(np.ascontiguousarray(validated.q_ic),np.array(list(asdict(self.settings).values()),dtype=np.float64))
        if not self.handle:raise MemoryError('Cannot allocate native Mahony')
        self.info=np.zeros(6,dtype=np.float64)
    def close(self):
        if getattr(self, "handle", None):self.lib.ego2_mahony_delete(self.handle);self.handle=None
    def __del__(self):self.close()
    @property
    def status(self):return status_text(self.info)
    @property
    def segment(self):return int(self.info[1])
    @property
    def resets(self):return int(self.info[2])
    @property
    def sparse_gaps(self):return int(self.info[3])
    @property
    def valid_samples(self):return int(self.info[5])
    @property
    def reference(self):
        state=np.empty(11);initialized=self.lib.ego2_mahony_state(self.handle,state)
        return state[4:8].copy() if initialized else None
    def reset(self,reason='',*,count=True):
        code=6 if 'backlog' in reason else 3
        self.lib.ego2_mahony_reset(self.handle,code,int(count));self.lib.ego2_mahony_info(self.handle,self.info)
    def update_batch(self,rows):
        rows=np.ascontiguousarray(rows,dtype=np.float64)
        if rows.ndim!=2 or rows.shape[1]!=8:raise ValueError('Expected [host, device, ax, ay, az, gx, gy, gz]')
        q=np.empty((len(rows),4));info=np.empty((len(rows),6))
        if self.lib.ego2_mahony_batch(self.handle,rows,len(rows),q,info):raise RuntimeError('Native Mahony arithmetic failure')
        if len(info):self.info=info[-1].copy()
        return q,info
    def update(self,stamp,acc,gyro):
        q,_=self.update_batch([[stamp,stamp,*acc,*gyro]])
        return q[0].copy() if np.isfinite(q[0]).all() else None

if __name__=='__main__':build()
