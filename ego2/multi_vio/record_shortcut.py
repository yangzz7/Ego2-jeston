"""Foreground Rerun Enter shortcut for X11, isolated from capture processes.

XI2 observes events without grabbing keys or changing another application's
input. Only Enter is inspected; no typed text is collected or saved. Xlib's
connection/error handler lives in this helper process, never in the VIO loop.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import select
import subprocess
import sys

from .storage import write_json


def belongs_to(pid, owner):
    """Also handles the CLI wrapper's child process and detached viewer sessions."""
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == owner:
            return True
        seen.add(pid)
        try:
            # comm can contain spaces and parentheses; ppid follows the last ')'.
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


class Cookie(C.Structure):
    _fields_ = [('type', C.c_int), ('serial', C.c_ulong), ('send_event', C.c_int),
                ('display', C.c_void_p), ('extension', C.c_int), ('evtype', C.c_int),
                ('cookie', C.c_uint), ('data', C.c_void_p)]


class Event(C.Union):
    _fields_ = [('cookie', Cookie), ('pad', C.c_long * 24)]


class RawKey(C.Structure):
    # Only the prefix through flags is read; valuators are not used.
    _fields_ = [('type', C.c_int), ('serial', C.c_ulong), ('send_event', C.c_int),
                ('display', C.c_void_p), ('extension', C.c_int), ('evtype', C.c_int),
                ('time', C.c_ulong), ('deviceid', C.c_int), ('sourceid', C.c_int),
                ('detail', C.c_int), ('flags', C.c_int)]


class EventMask(C.Structure):
    _fields_ = [('deviceid', C.c_int), ('mask_len', C.c_int),
                ('mask', C.POINTER(C.c_ubyte))]


class X11Enter:
    def __init__(self, owner, allow_external=False):
        self.owner, self.allow_external = owner, allow_external
        self.x = C.CDLL('libX11.so.6')
        self.xi = C.CDLL('libXi.so.6')
        P, U, I = C.c_void_p, C.c_ulong, C.c_int
        self.bind(self.x, 'XOpenDisplay', P, [C.c_char_p])
        self.bind(self.x, 'XCloseDisplay', I, [P])
        self.bind(self.x, 'XDefaultRootWindow', U, [P])
        self.bind(self.x, 'XConnectionNumber', I, [P])
        self.bind(self.x, 'XInternAtom', U, [P, C.c_char_p, I])
        self.bind(self.x, 'XGetWindowProperty', I, [P, U, U, C.c_long, C.c_long, I, U,
                  C.POINTER(U), C.POINTER(I), C.POINTER(U), C.POINTER(U), C.POINTER(P)])
        self.bind(self.x, 'XQueryExtension', I, [P, C.c_char_p, C.POINTER(I), C.POINTER(I), C.POINTER(I)])
        self.bind(self.x, 'XKeysymToKeycode', C.c_ubyte, [P, U])
        self.bind(self.x, 'XQueryPointer', I, [P, U, C.POINTER(U), C.POINTER(U),
                  C.POINTER(I), C.POINTER(I), C.POINTER(I), C.POINTER(I), C.POINTER(C.c_uint)])
        self.bind(self.x, 'XPending', I, [P])
        self.bind(self.x, 'XNextEvent', I, [P, C.POINTER(Event)])
        self.bind(self.x, 'XGetEventData', I, [P, C.POINTER(Cookie)])
        self.bind(self.x, 'XFreeEventData', None, [P, C.POINTER(Cookie)])
        self.bind(self.x, 'XFree', I, [P])
        self.bind(self.x, 'XFlush', I, [P])
        self.bind(self.xi, 'XIQueryVersion', I, [P, C.POINTER(I), C.POINTER(I)])
        self.bind(self.xi, 'XISelectEvents', I, [P, U, C.POINTER(EventMask), I])
        # A foreground window can close between lookup and property reads.
        self.error_handler = C.CFUNCTYPE(I, P, P)(lambda display, error: 0)
        self.bind(self.x, 'XSetErrorHandler', P, [type(self.error_handler)])
        self.x.XSetErrorHandler(self.error_handler)
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError('无法连接 X11 显示服务')
        self.root = self.x.XDefaultRootWindow(self.display)
        self.atoms = {name: self.x.XInternAtom(self.display, name.encode(), False)
                      for name in ('_NET_ACTIVE_WINDOW', '_NET_WM_PID', 'WM_CLASS')}
        self.opcode, event, error = I(), I(), I()
        if not self.x.XQueryExtension(self.display, b'XInputExtension', C.byref(self.opcode), C.byref(event), C.byref(error)):
            raise RuntimeError('桌面不支持 XInput2')
        major, minor = I(2), I(0)
        if self.xi.XIQueryVersion(self.display, C.byref(major), C.byref(minor)):
            raise RuntimeError('桌面不支持 XInput2 原始按键事件')
        mask = (C.c_ubyte * 2)(0, 1 << (13 - 8))  # XI_RawKeyPress
        subscription = EventMask(1, len(mask), mask)  # XIAllMasterDevices
        self.xi.XISelectEvents(self.display, self.root, C.byref(subscription), 1)
        self.x.XFlush(self.display)
        self.keys = {self.x.XKeysymToKeycode(self.display, sym) for sym in (0xff0d, 0xff8d)} - {0}

    @staticmethod
    def bind(lib, name, result, arguments):
        function = getattr(lib, name)
        function.restype, function.argtypes = result, arguments

    def property(self, window, name):
        actual, count, remaining = C.c_ulong(), C.c_ulong(), C.c_ulong()
        fmt, data = C.c_int(), C.c_void_p()
        result = self.x.XGetWindowProperty(self.display, window, self.atoms[name], 0, 256,
            False, 0, C.byref(actual), C.byref(fmt), C.byref(count), C.byref(remaining), C.byref(data))
        try:
            if result or not data or not count.value:
                return None
            if fmt.value == 8:
                return C.string_at(data, count.value)
            if fmt.value == 32:
                return C.cast(data, C.POINTER(C.c_ulong))[0]
            return None
        finally:
            if data: self.x.XFree(data)

    def foreground_matches(self):
        window = self.property(self.root, '_NET_ACTIVE_WINDOW')
        if not window:
            return False
        classes = self.property(window, 'WM_CLASS') or b''
        if b'rerun' not in classes.lower().split(b'\0'):
            return False
        pid = self.property(window, '_NET_WM_PID')
        return bool(pid and (self.allow_external or belongs_to(pid, self.owner)))

    def unmodified(self):
        root, child = C.c_ulong(), C.c_ulong()
        rx, ry, wx, wy, mask = C.c_int(), C.c_int(), C.c_int(), C.c_int(), C.c_uint()
        found = self.x.XQueryPointer(self.display, self.root, C.byref(root), C.byref(child),
            C.byref(rx), C.byref(ry), C.byref(wx), C.byref(wy), C.byref(mask))
        # Allow Caps Lock and Num Lock; ignore Shift/Ctrl/Alt/Super/AltGr chords.
        return bool(found) and not mask.value & (1 | 4 | 8 | 32 | 64 | 128)

    def triggers(self, key, flags):
        return (key in self.keys and not flags & (1 << 16) and  # XIKeyRepeat
                self.unmodified() and self.foreground_matches())

    def run(self, channel):
        write_json(channel/'shortcut.json', dict(ready=True))
        fd = self.x.XConnectionNumber(self.display)
        while os.getppid() == self.owner:
            if not self.x.XPending(self.display):
                select.select([fd], [], [], .5)
            while self.x.XPending(self.display):
                event = Event()
                self.x.XNextEvent(self.display, C.byref(event))
                cookie = event.cookie
                if cookie.type != 35 or cookie.extension != self.opcode.value:
                    continue
                if not self.x.XGetEventData(self.display, C.byref(cookie)):
                    continue
                try:
                    if cookie.evtype == 13:
                        key = C.cast(cookie.data, C.POINTER(RawKey)).contents
                        if self.triggers(key.detail, key.flags):
                            (channel/'open-panel').touch()
                finally:
                    self.x.XFreeEventData(self.display, C.byref(cookie))


class ViewerShortcut:
    """Small file mailbox; no Xlib calls or blocking waits in capture.poll()."""
    def __init__(self, channel, args):
        self.channel = Path(channel)
        self.process = None
        self.reported = False
        self.exit_reported = False
        supported = (sys.platform.startswith('linux') and os.environ.get('DISPLAY') and
                     not os.environ.get('WAYLAND_DISPLAY') and
                     not getattr(args, 'web', False) and not getattr(args, 'no_viz', False))
        if supported:
            command = [sys.executable, '-m', 'ego2.multi_vio.record_shortcut',
                       '--channel', str(channel), '--owner', str(os.getpid())]
            if getattr(args, 'connect', None): command.append('--external')
            try:
                self.process = subprocess.Popen(command, start_new_session=True)
            except OSError as exc:
                print(f'[record] 窗口快捷键启动失败：{exc}；仍可在启动终端按回车。', flush=True)

    def poll(self):
        status = self.channel/'shortcut.json'
        if not self.reported and status.exists():
            self.reported = True
            message = json.loads(status.read_text())
            if message.get('ready'):
                print('[record] 快捷键已就绪：Rerun 前台按 Enter 打开录制设置；录制中按 Enter 停止并保存。', flush=True)
            else:
                print(f'[record] 窗口快捷键不可用：{message.get("error")}；仍可在启动终端按回车。', flush=True)
        if self.process and self.process.poll() is not None and not self.exit_reported:
            self.exit_reported = True
            print('[record] 窗口快捷键进程已退出；仍可在启动终端按回车打开录制设置。', flush=True)
        request = self.channel/'open-panel'
        if request.exists():
            request.unlink(missing_ok=True)
            return True
        return False

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', required=True, type=Path)
    parser.add_argument('--owner', required=True, type=int)
    parser.add_argument('--external', action='store_true')
    args = parser.parse_args()
    try:
        listener = X11Enter(args.owner, args.external)
        try:
            listener.run(args.channel)
        finally:
            listener.x.XCloseDisplay(listener.display)
    except Exception as exc:
        write_json(args.channel/'shortcut.json', dict(ready=False, error=str(exc)))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
