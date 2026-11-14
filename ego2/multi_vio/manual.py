"""Nonblocking viewer/terminal/panel control, separate from camera lifetimes."""
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time

from ..common.paths import OUTPUT
from .selection import qt_environment
from .storage import FOLDERS, status, write_json
from .record_shortcut import ViewerShortcut


def destination(mode, directory='', name='', *, now=None):
    base = Path(directory.strip()).expanduser() if directory.strip() else OUTPUT
    if not base.is_absolute():
        raise ValueError('存储根目录请使用绝对路径，或通过“浏览”选择。')
    name = name.strip() or (now or datetime.now()).strftime('%Y-%m-%d_%H-%M-%S-%f')
    if name in ('.', '..') or any(c in name for c in '/\\\0') or any(ord(c) < 32 for c in name):
        raise ValueError('文件夹名称不能包含路径分隔符、控制字符或使用 .、..。')
    return base.resolve() / FOLDERS[mode] / name


def create_destination(mode, directory, name):
    target = destination(mode, directory, name)
    if target.exists() or target.is_symlink():
        raise ValueError(f'文件夹已存在，请更换名称：{target}')
    parent = target.parent
    ancestor = parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
        raise ValueError(f'保存位置不可写：{ancestor}')
    if shutil.disk_usage(ancestor).free < 2 * 1024**3:
        raise ValueError('保存位置可用空间不足 2 GiB。')
    parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return target


class ManualController:
    """Repeatable recordings; preparation and finalization never stop preview.

    prepare(root, token), ready()->bool, commit(t0), abort() are owned by the
    active mode. A separate Qt process never blocks the capture event loop.
    """
    def __init__(self, mode, args, prepare, ready, commit, abort, *, stdin=None,
                 stop=None, stopped=None, finalize=None):
        self.mode, self.args = mode, args
        self.prepare, self.ready, self.commit, self.abort = prepare, ready, commit, abort
        self.stop_recording, self.stopped, self.finalize = stop, stopped, finalize
        self.stopping = False
        self.stop_time = None
        self.finish_future = None
        self.finish_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ego2-record-finalize')
        self.saved = []
        self.stdin = sys.stdin if stdin is None else stdin
        self.temporary = tempfile.TemporaryDirectory(prefix='ego2-record-panel-')
        self.directory = Path(self.temporary.name)
        self.process = None
        self.completion_process = None
        self.saving_process = None
        self.save_started = time.monotonic()
        self.last_progress = (None, 0.)
        self.root = None
        self.pending = False
        self.active = False
        self.request_id = None
        self.input_enabled = True
        self.started_at = None
        self.shortcut = ViewerShortcut(self.directory, args)
        print('[record] 当前仅预览；Rerun 窗口快捷键就绪后可直接按回车打开录制设置。'
              '也可在启动终端按回车；Ctrl+C 退出。', flush=True)

    def open(self):
        if self.stopping:
            return  # Held/repeated Enter cannot start another recording while saving.
        if self.active:
            if self.stop_recording is None:
                raise RuntimeError('当前录制模式缺少停止回调')
            self.stopping = True
            # Give each independently running writer time to receive the same
            # end boundary. Preview continues throughout draining/finalization.
            self.stop_time = time.monotonic() + .15
            try:
                status(self.root, 'stopping', record_stop_monotonic=self.stop_time)
                self.stop_recording(self.stop_time)
            except Exception as exc:
                self.finish_failed(exc)
                return
            self.stop_deadline = time.monotonic() + 120
            self.save_started = time.monotonic()
            self.last_progress = (None, 0.)
            self.report_progress('等待采集缓存写入完成',0,0)
            self.show_saving()
            print(f'[record] 已请求停止录制，正在保存：{self.root}；继续预览。', flush=True)
            return
        if self.completion_process and self.completion_process.poll() is None:
            (self.directory/'raise-completion').touch()
            return
        if self.process and self.process.poll() is None:
            (self.directory/'raise-panel').touch()
            return
        for name in ('request.json', 'response.json', 'raise-panel'):
            (self.directory/name).unlink(missing_ok=True)
        self.request_id = None
        command = [sys.executable, '-m', 'ego2.multi_vio.record_panel',
                   '--mode', self.mode, '--channel', str(self.directory)]
        if self.args.session:
            command += ['--preset', str(self.args.session)]
        self.process = subprocess.Popen(command, env=qt_environment(), start_new_session=True)

    def poll(self):
        if self.shortcut.poll():
            self.open()
        if self.input_enabled:
            try:
                if select.select([self.stdin], [], [], 0)[0]:
                    line = self.stdin.readline()
                    if line == '':
                        self.input_enabled = False
                    elif not line.strip():
                        self.open()
            except (OSError, ValueError, TypeError):
                self.input_enabled = False
        request = self.directory/'request.json'
        if request.exists() and not self.pending and not self.active and not self.stopping:
            message = json.loads(request.read_text())
            if message['id'] != self.request_id:
                self.request_id = message['id']
                try:
                    self.root = create_destination(self.mode, message.get('directory', ''), message.get('name', ''))
                    self.pending = True
                    self.stop_time = None
                    status(self.root, 'preparing', mode=self.mode)
                    self.deadline = time.monotonic() + 30
                    self.prepare(self.root, self.request_id)
                except Exception as exc:
                    self.fail(exc)
        if self.pending:
            try:
                if self.ready():
                    # Participants already have their writers open. The common
                    # future cutoff excludes in-flight pre-recording frames.
                    self.started_at = time.monotonic() + .35
                    self.commit(self.started_at)
                    status(self.root, 'recording', record_start_monotonic=self.started_at)
                    write_json(self.directory/'response.json', dict(id=self.request_id, ok=True, path=str(self.root)))
                    self.active, self.pending = True, False
                    print(f'[recording] {self.root} | 再按 Enter 停止录制并保存，预览继续；Ctrl+C 退出。', flush=True)
                elif time.monotonic() > self.deadline:
                    raise RuntimeError('录制准备超时，请检查各相机是否正常取流。')
            except Exception as exc:
                self.fail(exc)
        if self.stopping:
            try:
                if self.finish_future is None:
                    if self.stopped():
                        # Detach writers and snapshot mutable live state on this
                        # thread, then run only the detached finalizer in background.
                        finish = self.finalize()
                        status(self.root, 'finalizing')
                        self.finish_future = self.finish_pool.submit(finish)
                    elif time.monotonic() > self.stop_deadline:
                        raise RuntimeError('停止录制超时；保留缓存，未标记保存成功。')
                self.wait_finalization(block=False, notify=True)
            except Exception as exc:
                self.finish_failed(exc)
        if self.process and self.process.poll() is not None and not self.pending:
            if not self.active:
                print('[record] 录制面板已关闭，继续预览。', flush=True)
            self.process = None
        if self.completion_process and self.completion_process.poll() is not None:
            if self.completion_process.returncode:
                print(f'[record] 录制完成面板异常退出；数据已保存：{self.saved[-1]}', flush=True)
            self.completion_process = None

    def wait_finalization(self, *, block=True, notify=False):
        # Shutdown callers wait without opening a dialog that promises preview.
        if self.finish_future is None or (not block and not self.finish_future.done()):
            return
        try:
            self.finish_future.result()
        except Exception as exc:
            self.finish_failed(exc)
            return
        self.finish_future = None
        self.active = self.stopping = False
        self.saved.append(self.root)
        self.report_progress('保存完成',1,1,phase='complete')
        self.close_saving()
        suffix = '，继续预览；按 Enter 可录制下一份。' if notify else '。'
        print(f'[saved] {self.root} | 保存完成{suffix}', flush=True)
        if notify:
            self.show_completion(self.root)

    def report_progress(self, stage, current, total, *, phase='saving'):
        now = time.monotonic()
        previous, updated = self.last_progress
        if phase == 'saving' and stage == previous and now-updated < 1 and (not total or current < total):
            return
        self.last_progress = (stage,now)
        # Progress UI/transport failures cannot invalidate captured data.
        try:
            write_json(self.directory/'save-progress.json',dict(path=str(self.root),phase=phase,
                stage=stage,current=current,total=total,started=self.save_started))
        except OSError as exc:
            print(f'[save] 进度窗口更新失败：{exc}',flush=True)
        detail = f' {current} / {total}' if total else ''
        print(f'[save] {stage}{detail}；已用时 {now-self.save_started:.0f} 秒',flush=True)

    def show_saving(self):
        if (self.mode != 'gesture' or getattr(self.args,'headless',False)
                or not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))):
            return
        self.close_saving()
        try:
            self.saving_process = subprocess.Popen(
                [sys.executable,'-m','ego2.multi_vio.record_saving','--path',str(self.root),
                 '--channel',str(self.directory)],env=qt_environment(),start_new_session=True)
        except Exception as exc:
            print(f'[save] 无法打开进度窗口：{exc}；请查看终端进度。',flush=True)

    def close_saving(self):
        process,self.saving_process = self.saving_process,None
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()

    def show_completion(self, root):
        if getattr(self.args, 'headless', False) or not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            return
        # Notification failure must not invalidate a successfully saved dataset.
        try:
            (self.directory/'raise-completion').unlink(missing_ok=True)
            self.completion_process = subprocess.Popen(
                [sys.executable, '-m', 'ego2.multi_vio.record_complete',
                 '--path', str(root), '--channel', str(self.directory)],
                env=qt_environment(), start_new_session=True)
        except Exception as exc:
            print(f'[record] 无法打开录制完成面板：{exc}；数据已保存：{root}', flush=True)

    def finish_failed(self, exc):
        if self.finish_future is None:
            try:
                self.abort()
            except Exception as cleanup:
                print(f'[record] 停止写入失败：{cleanup}', flush=True)
        self.active = self.stopping = False
        self.finish_future = None
        # Stop/flush failures must never look like a complete dataset. Keep
        # caches for diagnosis; a subsequent prepare will cleanly detach them.
        try:
            status(self.root, 'incomplete', error=str(exc))
        except OSError as metadata_error:
            print(f'[record] 无法写入失败状态：{metadata_error}', flush=True)
        print(f'[record] 保存失败，缓存保留在 {self.root}：{exc}；继续预览。', flush=True)
        self.report_progress(str(exc),0,0,phase='error')

    def fail(self, exc):
        if self.pending:
            try:
                self.abort()
            except Exception as cleanup:
                print(f'[record] 清理准备过程失败：{cleanup}', file=sys.stderr, flush=True)
            try:
                status(self.root, 'incomplete', error=str(exc))
            except OSError as metadata_error:
                print(f'[record] 无法写入失败状态：{metadata_error}',file=sys.stderr,flush=True)
        self.pending = False
        write_json(self.directory/'response.json', dict(id=self.request_id, ok=False, error=str(exc)))
        print(f'[record] 未开始录制：{exc}', flush=True)

    def close(self):
        self.shortcut.close()
        self.wait_finalization()
        self.finish_pool.shutdown(wait=True)
        self.close_saving()
        if self.pending:
            self.fail(RuntimeError('程序在录制准备期间退出'))
        for process in (self.process, self.completion_process):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        self.temporary.cleanup()


class WorkerControl:
    """Per-worker file mailbox: prepare first, then accept a shared cutoff."""
    def __init__(self, directory, prepare=lambda root: None, abort=lambda: None,
                 finish=lambda: None):
        self.directory = Path(directory)
        self.prepare, self.abort = prepare, abort
        self.finish = finish
        self.finish_future = None
        self.stop_time = None
        self.stop_done = False
        self.token = None
        self.start_time = None
        self.phase = None

    def poll(self):
        if self.stop_time is not None and not self.stop_done and time.monotonic() >= self.stop_time:
            try:
                if self.finish_future is None:
                    self.finish_future = self.finish()
                    if self.finish_future is None:
                        self.stop_done = True
                if self.finish_future is not None and self.finish_future.done():
                    self.finish_future.result()
                    self.stop_done = True
                if self.stop_done:
                    write_json(self.directory/'state.json', dict(id=self.token, stopped=True))
            except Exception as exc:
                self.stop_done = True
                write_json(self.directory/'state.json', dict(id=self.token, stopped=False, error=str(exc)))
        path = self.directory/'command.json'
        if not path.exists():
            return
        message = json.loads(path.read_text())
        key = (message['id'], message['phase'])
        if key == self.phase:
            return
        self.phase = key
        if message['phase'] == 'prepare':
            try:
                self.abort()
                self.start_time = None
                self.stop_time = None
                self.stop_done = False
                self.finish_future = None
                self.prepare(Path(message['directory']))
                self.token = message['id']
                reply = dict(id=self.token, ready=True)
            except Exception as exc:
                reply = dict(id=message['id'], ready=False, error=str(exc))
            write_json(self.directory/'state.json', reply)
        elif message['phase'] == 'commit' and message['id'] == self.token:
            self.start_time = float(message['start_time'])
        elif message['phase'] == 'stop' and message['id'] == self.token:
            self.start_time = float(message.get('start_time', self.start_time))
            self.stop_time = float(message['stop_time'])
        elif message['phase'] == 'abort':
            self.start_time = None
            self.stop_time = None
            self.abort()


class WorkerGroup:
    def __init__(self, root, roles):
        self.directories = {r: Path(root)/r for r in roles}
        for path in self.directories.values(): path.mkdir(parents=True, exist_ok=True)
        self.token = None

    def prepare(self, root, token):
        self.token = token
        for role, directory in self.directories.items():
            work = Path(root)/'_work'/role
            work.mkdir(parents=True, exist_ok=True)
            write_json(directory/'command.json', dict(id=token, phase='prepare', directory=str(work)))

    def ready(self):
        for directory in self.directories.values():
            path = directory/'state.json'
            if not path.exists(): return False
            reply = json.loads(path.read_text())
            if reply.get('id') != self.token: return False
            if reply.get('error'): raise RuntimeError(reply['error'])
            if not reply.get('ready'): return False
        return True

    def commit(self, start):
        self.start_time = start
        for directory in self.directories.values():
            write_json(directory/'command.json', dict(id=self.token, phase='commit', start_time=start))

    def abort(self):
        if self.token is None: return
        for directory in self.directories.values():
            write_json(directory/'command.json', dict(id=self.token, phase='abort'))

    def stop(self, end):
        for directory in self.directories.values():
            write_json(directory/'command.json', dict(id=self.token, phase='stop',
                       start_time=self.start_time, stop_time=end))

    def stopped(self):
        for directory in self.directories.values():
            path = directory/'state.json'
            if not path.exists(): return False
            reply = json.loads(path.read_text())
            if reply.get('id') != self.token: return False
            if reply.get('error'): raise RuntimeError(reply['error'])
            if not reply.get('stopped'): return False
        return True
