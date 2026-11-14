"""Build only the standalone checkout's vendored ORB and SDK sources."""
from pathlib import Path
import subprocess
from ..common.paths import ROOT, BUILD

def native_is_stale(executable, cfg):
    if not executable.exists():
        return True
    roots = [ROOT/'ego2/reconstruction/native', ROOT/'ego2/multi_vio/native', ROOT/'ego2/common/native',
             Path(cfg['orb_source'])/'include', Path(cfg['orb_source'])/'src',
             Path(cfg['orb_source'])/'Thirdparty/usb/uvc']
    return any(p.stat().st_mtime > executable.stat().st_mtime
               for root in roots for p in root.rglob('*') if p.is_file())

def build_native(cfg, jobs=2):
    from ..common.cyper_cuda import capture_backend
    if capture_backend() == 'cuda':
        from ..common.build_capture_cuda import build_capture_cuda
        build_capture_cuda()
    source = Path(cfg["orb_source"]).resolve()
    native = Path(__file__).parent / "native"
    subprocess.run(["cmake","-S",str(native),"-B",str(BUILD/"cmake"),
                    f"-DORB_SOURCE={source}",f"-DEGO2_OUTPUT={BUILD}",
                    "-DCMAKE_BUILD_TYPE=Release"],check=True)
    subprocess.run(["cmake","--build",str(BUILD/"cmake"),"--parallel",str(jobs)],check=True)
    return BUILD/"bin/ego2_reconstruction_native"
