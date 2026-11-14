"""Local EGL for Jetson's headless inference child; UI stays in the parent."""
import json
import os
from pathlib import Path

from .recognizer import requested_backend


def prepare_worker_environment(settings):
    # Call only inside multi-camera's spawned gesture worker, before importing
    # MediaPipe. Leaving DISPLAY unset selects Jetson's local EGL display.
    # Do not restore it while the model's native GPU threads are alive.
    if not Path('/etc/nv_tegra_release').is_file() or requested_backend(settings) == 'cpu':
        return ()
    removed = tuple(key for key in ('DISPLAY', 'WAYLAND_DISPLAY') if key in os.environ)
    for key in removed:
        os.environ.pop(key)
    print('[gesture-egl] headless Jetson worker; local EGL; isolated ' +
          (', '.join(removed) if removed else 'no inherited display') +
          '; parent UI environment unchanged', flush=True)
    return removed


class BackendReporter:
    def __init__(self, path):
        self.path = Path(path)
        self.previous = None

    def __call__(self, recognizers):
        status = [dict(eye=index, backend=r.backend, requested=r.requested_backend,
                       fallbacks=r.fallbacks, reason=getattr(r, 'last_fallback', None))
                  for index, r in enumerate(recognizers)]
        if status == self.previous:
            return
        temporary = self.path.with_suffix('.tmp')
        try:
            temporary.write_text(json.dumps(status, ensure_ascii=False))
            temporary.replace(self.path)
        except OSError as exc:
            # A diagnostic file failure cannot stop inference/recording.
            print(f'[gesture-inference] backend status unavailable: {exc}', flush=True)
        self.previous = status


def display_backend_status(path, role, previous):
    try:
        status = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return previous
    if status != previous:
        values = ', '.join(f'eye{s["eye"]}={s["backend"].upper()} '
                           f'(requested={s["requested"]}, fallbacks={s["fallbacks"]})'
                           for s in status)
        print(f'[gesture-inference] {role}: {values}', flush=True)
        for s in status:
            if s.get('reason'):
                print(f'[gesture-inference] {role} eye{s["eye"]}: {s["reason"]}', flush=True)
    return status
