"""Offline CPU/CUDA pixel checks and a bounded four-camera ISP benchmark.

Reads no camera. JSON measurements include CPU time, wall latency and the
background-load caveat; neither is presented as full application frame rate.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from cyperstereo_isp_portable.python.cyperstereo_isp import Isp


def frame(version, kind, seed=9):
    rng=np.random.default_rng(seed)
    y,x=np.indices((1024,1280))
    if kind=='noise':
        raw=rng.integers(0,256,(1024,1280,2),dtype=np.uint8)
    elif kind=='dark':
        raw=rng.integers(8,29,(1024,1280,2),dtype=np.uint8)
    else:
        base=24+((x//160+2*(y//128))*17)%190
        raw=np.stack((base,255-base),axis=-1).astype(np.uint8)
        raw[:256,:,0]=120+(x[:256]%2)*45-(y[:256]%2)*25
        raw[256:512,:,1]=64+((x[256:512]+y[256:512])%2)*110
    raw[-1]=0
    words=raw[-1].reshape(-1).view('<u2')
    words[0]=2;words[1]=version;words[4]=10
    if version>=4:
        exp,temp,gain=(123,127,131) if version==5 else (69,73,77)
        words[exp:exp+2]=100;words[temp:temp+2]=149*256;words[gain:gain+2]=4*64
    return raw


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--library',required=True);p.add_argument('--cpu-library',required=True)
    p.add_argument('--output',required=True);p.add_argument('--rounds',type=int,default=6)
    a=p.parse_args();cv2.setNumThreads(1)
    config=Path(__file__).resolve().parents[1]/'cyperstereo_isp_portable/configs/s200032.yaml'
    report={'pixels':[],'benchmarks':[],'scope':'Offline synthetic full-resolution stereo ISP, not application FPS; other processes may be running.'}
    with Isp(config,library=a.cpu_library) as baseline, Isp(config,library=a.library,backend='cpu') as cpu, Isp(config,library=a.library,backend='cuda') as gpu:
        for version in range(3,8):
            for kind in ('pattern','noise','dark'):
                raw=frame(version,kind);before=hashlib.sha256(raw).hexdigest()
                baseline.reset();cpu.reset();gpu.reset()
                original=baseline.process_transport(raw); optimized=cpu.process_transport(raw); accelerated=gpu.process_transport(raw)
                assert before==hashlib.sha256(raw).hexdigest(),'Input was modified'
                for x,y in zip(original,optimized):np.testing.assert_array_equal(x,y)
                np.testing.assert_allclose(baseline.last_white_balance_rgb,gpu.last_white_balance_rgb,rtol=0,atol=0)
                delta=np.abs(np.stack(original).astype(np.int16)-np.stack(accelerated).astype(np.int16))
                item={'firmware':version,'pattern':kind,'mean_abs':float(delta.mean()),'p999':float(np.quantile(delta,.999)),'max_abs':int(delta.max())}
                report['pixels'].append(item);print(json.dumps(item),flush=True)
                # A strict display-only tolerance; never use this output for algorithms/recording.
                assert item['mean_abs']<=.12 and item['p999']<=2 and item['max_abs']<=8, item
        for number in range(4):
            raw=frame(7,'pattern',number)
            raw[:-1]=np.clip(raw[:-1].astype(np.int16)+number*4,0,255).astype(np.uint8)
            expected=baseline.process_transport(raw);actual=gpu.process_transport(raw)
            np.testing.assert_array_equal(baseline.last_white_balance_rgb,gpu.last_white_balance_rgb)
            assert np.abs(np.stack(expected).astype(int)-np.stack(actual).astype(int)).mean()<=.12
        raw=frame(8,'pattern')
        try:gpu.process_transport(raw)
        except RuntimeError:pass
        else:raise AssertionError('Invalid firmware accepted')
    raw=frame(7,'pattern')
    for backend in ('baseline','cpu','cuda'):
        workers=[Isp(config,library=a.cpu_library) if backend=='baseline' else Isp(config,library=a.library,backend=backend) for _ in range(4)]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda isp:isp.process_transport(raw),workers))
                durations=[];cpu0=time.process_time();wall0=time.perf_counter()
                for _ in range(a.rounds):
                    t=time.perf_counter();list(pool.map(lambda isp:isp.process_transport(raw),workers));durations.append(time.perf_counter()-t)
                cpu_seconds=time.process_time()-cpu0;wall_seconds=time.perf_counter()-wall0
                item={'backend':backend,'cameras':4,'rounds':a.rounds,'wall_ms_per_four_pairs':1000*wall_seconds/a.rounds,
                      'cpu_ms_per_pair':1000*cpu_seconds/(a.rounds*4),'wall_p50_ms':1000*float(np.median(durations)),
                      'average_cpu_cores':cpu_seconds/wall_seconds}
                report['benchmarks'].append(item);print(json.dumps(item),flush=True)
        finally:
            for isp in workers:isp.close()
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print('CPU/CUDA pixel, input immutability, firmware, AWB and four-camera ISP checks passed.',flush=True)

if __name__=='__main__':main()
