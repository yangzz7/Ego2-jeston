"""Build the shared live RAW converter on Jetson (CUDA toolkit required)."""
import os
from pathlib import Path
import subprocess
from .cyper_cuda import ROOT


def build_capture_cuda():
    source = ROOT / "ego2/common/native"
    output = ROOT / "build/cyper_capture"
    nvcc = os.environ.get("CUDACXX", "/usr/local/cuda/bin/nvcc")
    subprocess.run(["cmake", "-S", str(source), "-B", str(output),
                    f"-DCMAKE_CUDA_COMPILER={nvcc}", "-DCMAKE_CUDA_ARCHITECTURES=87",
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(output), "--parallel", "2"], check=True)


if __name__ == "__main__":
    build_capture_cuda()
