import json,time,os,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
from tests.check_display_cuda import frame
from cyperstereo_isp_portable.python.cyperstereo_isp import Isp
cv2.setNumThreads(1)
parser=argparse.ArgumentParser()
parser.add_argument('--before',required=True)
parser.add_argument('--after',default='build/cyperstereo_isp_display/libcyperstereo_isp_c.so')
parser.add_argument('--report',type=Path)
args=parser.parse_args()
old,new=args.before,args.after
report={'pixels':[],'benchmarks':{},'scope':'Offline GPU ISP and JPEG; not application FPS'}
config='cyperstereo_isp_portable/configs/s200097.yaml'
with Isp(config,library=old,backend='cuda') as a,Isp(config,library=new,backend='cuda') as b:
    for version in range(3,8):
        for kind in ['pattern','noise','dark']:
            raw=frame(version,kind);snap=raw.copy();a.reset();b.reset()
            for _ in range(3):
                x=a.process_transport(raw);y=b.process_transport(raw)
                for p,q in zip(x,y):np.testing.assert_array_equal(p,q)
                np.testing.assert_array_equal(a.last_white_balance_rgb,b.last_white_balance_rgb)
                np.testing.assert_array_equal(raw,snap)
            report['pixels'].append({'firmware':version,'kind':kind,'frames':3,'exact':True,'awb_exact':True,'raw_unchanged':True})
    raw=frame(7,'pattern');
    for backend,isp in [('before',a),('after',b)]:
        for _ in range(4):isp.process_transport(raw)
        timings=[];cpu=time.process_time()
        for _ in range(25):
            start=time.perf_counter();images=isp.process_transport(raw);timings.append(1000*(time.perf_counter()-start))
        report['benchmarks'][backend+'_isp']={'mean_ms':float(np.mean(timings)),'p95_ms':float(np.quantile(timings,.95)),'cpu_ms':1000*(time.process_time()-cpu)/25}
    def jpeg(image):return cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90])[1]
    with ThreadPoolExecutor(2) as pool:
        for name,run in [('serial',lambda:list(map(jpeg,images))),('parallel',lambda:list(pool.map(jpeg,images)))]:
            for _ in range(3):run()
            values=[]
            for _ in range(25):
                start=time.perf_counter();run();values.append(1000*(time.perf_counter()-start))
            report['benchmarks'][name+'_jpeg']={'mean_ms':float(np.mean(values)),'p95_ms':float(np.quantile(values,.95))}
        for x,y in zip(list(map(jpeg,images)),list(pool.map(jpeg,images))):np.testing.assert_array_equal(x,y)
if args.report:args.report.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
