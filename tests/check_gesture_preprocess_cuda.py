import json,time,os,argparse
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from ego2.gesture.preprocess import CudaOperation,WhiteBalance
from ego2.common.sources import white_balance
from ego2.common.config import orient
from ego2.multi_vio.config import MULTI_CONFIG
from ego2.multi_vio.modes import load_mode_config
from ego2.gesture.rectification import Rectifier
parser=argparse.ArgumentParser()
parser.add_argument('--report',type=Path)
args=parser.parse_args()
cv2.setNumThreads(1)
rng=np.random.default_rng(52);report={'pixel_checks':[],'benchmarks':{},'scope':'offline exact pixel checks and timings; not camera FPS'}
for shape in [(9,11,3),(480,640,3),(1024,1280,3)]:
    src=rng.integers(0,256,shape,dtype=np.uint8)
    before=src.copy()
    lut=rng.integers(0,256,(3,256),dtype=np.uint8)
    op=CudaOperation(shape)
    actual=op.run(src,lut)
    np.testing.assert_array_equal(actual,cv2.merge([cv2.LUT(src[:,:,c],lut[c]) for c in range(3)]))
    saved=actual.copy();op.run(np.zeros_like(src),lut);np.testing.assert_array_equal(actual,saved);op.close()
    mx=rng.uniform(-2,shape[1]+2,(240,320)).astype('float32');my=rng.uniform(-2,shape[0]+2,(240,320)).astype('float32')
    # Include fractions exactly at half quantization and outside all borders.
    mx[0,:]=np.arange(320)/32+.5/32;my[0,:]=np.arange(320)/32+.5/32
    op=CudaOperation(shape,(mx,my));got=op.run(src)
    expected=cv2.cvtColor(cv2.remap(src,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT),cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(got,expected);np.testing.assert_array_equal(src,before);op.close()
    report['pixel_checks'].append({'shape':shape,'lut_exact':True,'remap_rgb_exact':True,'input_immutable':True,'output_owned':True})
op=CudaOperation((1024,1280,3))
for seed in range(20):
    im=np.random.default_rng(seed).integers(0,256,(1024,1280,3),dtype=np.uint8)
    np.testing.assert_array_equal(op.sample(im),cv2.resize(im,(160,120),interpolation=cv2.INTER_AREA))
op.close()
report['pixel_checks'].append({'area_sample_exact_random_frames':20})
wb=WhiteBalance()
for kind in ['noise','black','white','red','gradient']:
    image=rng.integers(0,256,(1024,1280,3),dtype=np.uint8)
    if kind=='black':image.fill(0)
    if kind=='white':image.fill(255)
    if kind=='red':image[:]=[0,0,255]
    if kind=='gradient':image[:]=np.arange(1280,dtype=np.uint8)[None,:,None]
    np.testing.assert_array_equal(wb(image),white_balance(image))
    report['pixel_checks'].append({'white_balance':kind,'exact':True})
image=rng.integers(0,256,(1024,1280,3),dtype=np.uint8)
def bench(name,f,n=25):
    for _ in range(4):f()
    t=time.process_time();values=[]
    for _ in range(n):
        start=time.perf_counter();f();values.append((time.perf_counter()-start)*1000)
    report['benchmarks'][name]={'mean_ms':float(np.mean(values)),'p95_ms':float(np.quantile(values,.95)),'cpu_ms':(time.process_time()-t)*1000/n}
bench('white_balance_cpu',lambda:white_balance(image));bench('white_balance_cuda',lambda:wb(image))
for serial in ['s200032','s200097','s200098','s200100']:
    cfg,devices=load_mode_config(MULTI_CONFIG,mode='gesture',selection={'left_hand':serial})
    p=devices['left_hand']['profile'];cams=devices['left_hand']['cameras']
    for c in p['channels']:
        with patch.dict(os.environ,EGO2_GESTURE_PREPROCESS_BACKEND='cpu'):cpu=Rectifier(cams[c['camera']],c,p['inference_view'])
        with patch.dict(os.environ,EGO2_GESTURE_PREPROCESS_BACKEND='cuda'):gpu=Rectifier(cams[c['camera']],c,p['inference_view'])
        np.testing.assert_array_equal(cpu.prepare(image),gpu.prepare(image))
        assert gpu.accelerator.operation is not None and not gpu.accelerator.failed
        report['pixel_checks'].append({'serial':serial,'eye':c['camera'],'calibration_exact':True})
        if serial=='s200097':
            bench(c['camera']+'_prepare_cpu',lambda:cpu.prepare(image));bench(c['camera']+'_prepare_cuda',lambda:gpu.prepare(image))
        cpu.close();gpu.close()
assert wb.accelerator.operation is not None and not wb.accelerator.failed
wb.close()
if args.report:args.report.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
