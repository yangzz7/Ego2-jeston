"""Prepare/play one session off the Qt thread, with a cancellable process group."""
import argparse
import subprocess
import time
from pathlib import Path

from .session_catalog import inspect_session
from .storage import write_json, validate_rrd


def run(root, workspace):
    workspace = Path(workspace)
    state = workspace/'state.json'
    viewer = None
    try:
        item = inspect_session(root)
        if not item.playable: raise ValueError(item.error)
        if item.format == 'LeRobot V3':
            from .replay import export_dataset
            def progress(count, total):
                write_json(state, dict(phase='preparing', text=f'正在准备回放… {count} / {total} 帧'))
            path = export_dataset(item.path, workspace/'replay.rrd', progress=progress)
        else:
            write_json(state, dict(phase='preparing', text='正在检查回放文件…'))
            path = item.path/'replay.rrd'
            validate_rrd(path)
        write_json(state, dict(phase='preparing', text='正在启动 Rerun…'))
        # Own a separate Viewer even if a live acquisition window still exists;
        # its lifetime determines when the temporary RRD may be removed.
        viewer = subprocess.Popen(['rerun', str(path), '--new'])
        # Catch immediate native Viewer launch failures before hiding the panel.
        deadline = time.monotonic()+1.
        while time.monotonic() < deadline and viewer.poll() is None:
            time.sleep(.05)
        if viewer.poll() not in (None, 0):
            raise RuntimeError(f'Rerun 启动失败（退出码 {viewer.returncode}），请查看终端日志。')
        write_json(state, dict(phase='playing', text='Rerun 已启动'))
        code = viewer.wait()
        if code: raise RuntimeError(f'Rerun 异常退出（退出码 {code}）。')
        return 0
    except Exception as exc:
        write_json(state, dict(phase='error', error=str(exc)))
        print(f'[replay] {exc}', flush=True)
        return 1
    finally:
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
            try: viewer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                viewer.kill(); viewer.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True)
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()
    return run(args.session, args.workspace)


if __name__ == '__main__':
    raise SystemExit(main())
