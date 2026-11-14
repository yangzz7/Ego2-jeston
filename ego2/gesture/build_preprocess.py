"""Build exact gesture preprocessing on Jetson, independently of RAW capture."""
import os
from pathlib import Path
import subprocess


def build():
    root=Path(__file__).resolve().parents[2]
    out=root/'build/gesture_preprocess'
    subprocess.run(['cmake','-S',str(root/'ego2/gesture/native'),'-B',str(out),
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=87',
        '-DCMAKE_CUDA_COMPILER='+os.environ.get('CUDACXX','/usr/local/cuda/bin/nvcc')],check=True)
    subprocess.run(['cmake','--build',str(out),'--parallel','2'],check=True)

if __name__=='__main__': build()
