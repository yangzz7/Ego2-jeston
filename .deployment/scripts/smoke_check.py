"""Target-side offline checks. No physical camera is opened."""
import importlib,json,os,subprocess,sys,tempfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
report={'hardware_tested':False,'desktop_rendering_tested':False,'checks':{}}
for name in ['cv2','numpy','yaml','mediapipe','zmq','pyarrow','pandas','rerun','datafusion','PySide6']:
    module=importlib.import_module(name)
    report['checks'][name]=getattr(module,'__version__','imported')
import cv2,numpy as np
assert hasattr(cv2.aruco,'ArucoDetector')
from ego2.gesture.config import load_config
from ego2.gesture.recognizer import Recognizer
cfg,_=load_config(root/'configs/gesture.yaml',sn='s200032')
recognizer=Recognizer(cfg['gesture'])
try:
    recognizer.detect(np.zeros((640,640,3),dtype=np.uint8),0)
    report['checks']['gesture_backend']=dict(requested=recognizer.requested_backend, actual=recognizer.backend, fallbacks=recognizer.fallbacks)
finally:recognizer.close()
report['checks']['gesture_model']='Loaded real model and processed one black frame; not an accuracy test'
from ego2.gesture.preflight import _ffmpeg
report['checks']['ffmpeg']=_ffmpeg()
# Import/build actual dialogs in a separate offscreen Qt process so cv2's
# bundled Qt5 plugin environment cannot contaminate Qt6.
env=os.environ.copy();env['QT_QPA_PLATFORM']='offscreen'
env.pop('QT_QPA_PLATFORM_PLUGIN_PATH',None);env.pop('QT_QPA_FONTDIR',None)
subprocess.run([sys.executable,'-c',
    'from PySide6.QtWidgets import QApplication; app=QApplication([]); '
    'import ego2.multi_vio.startup, ego2.multi_vio.record_panel, ego2.multi_vio.record_complete; '
    'print("Qt GUI modules imported in offscreen mode")'],cwd=root,env=env,check=True)
report['checks']['qt_offscreen']='passed; actual Xorg/Rerun rendering still pending'
for binary in ['ego2_reconstruction_native','ego2_multi_vio_native']:
    path=root/'build/reconstruction/bin'/binary
    if not path.is_file():raise RuntimeError(f'Missing native executable: {path}')
    result=subprocess.run(['ldd',str(path)],capture_output=True,text=True,check=True)
    if 'not found' in result.stdout:raise RuntimeError(result.stdout)
    report['checks'][binary]='dynamic dependencies resolved'
with tempfile.TemporaryDirectory(prefix='ego2-offline-check-') as temp:
    subprocess.run([sys.executable,'-m','ego2.aprilgrid','--demo','--headless','--frames','3',
                    '--session',str(Path(temp)/'aprilgrid')],cwd=root,check=True)
report['checks']['aprilgrid_synthetic']='3 frames; no physical camera'
report['passed']=True
destination=root/'.deployment/smoke-report.json';destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
