"""One Rerun connection, recording and local web-server lifecycle for both apps."""
from pathlib import Path
import os
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import quote
import rerun as rr

TIMELINE = "capture_time"

class ViewerConnection:
    def _configure(self, profile, headless, save, connect, web, web_port, grpc_port):
        no_desktop = sys.platform.startswith("linux") and not any(
            os.environ.get(key) for key in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET"))
        if web or (no_desktop and not headless and not connect):
            connect = self._start_web(profile, web_port, grpc_port)
        if save and (not headless or connect):
            # Use set_sinks: calling save() after spawn/connect replaces the sink.
            sinks = [rr.FileSink(str(save))]
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            if not connect:
                self.recording.spawn(connect=False, memory_limit=profile["viewer"]["memory_limit"],
                                     hide_welcome_screen=True)
            sinks.append(rr.GrpcSink(connect or "rerun+http://127.0.0.1:9876/proxy"))
            self.recording.set_sinks(*sinks)
        elif save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            self.recording.save(save)
        elif connect:
            self.recording.connect_grpc(connect)
        elif not headless:
            self.recording.spawn(memory_limit=profile["viewer"]["memory_limit"],
                                 hide_welcome_screen=True)
        else:
            raise ValueError("Headless visualization requires --save")
        self.recording.send_blueprint(self.blueprint)


    def _start_web(self, profile, web_port, grpc_port):
        executable = shutil.which("rerun")
        if executable is None:
            raise RuntimeError("rerun CLI is missing from PATH; install requirements.txt")
        # Never accidentally reuse another application's occupied port.
        for port in (web_port, grpc_port):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError as exc:
                    raise RuntimeError(f"Port {port} is unavailable; set --web-port/--grpc-port or use --connect") from exc
        self.web_process = subprocess.Popen([
            executable, "--serve-web", "--bind", "127.0.0.1",
            "--port", str(grpc_port), "--web-viewer-port", str(web_port),
            "--server-memory-limit", profile["viewer"]["memory_limit"],
        ])
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if self.web_process.poll() is not None:
                    raise RuntimeError("Rerun web server exited during startup")
                try:
                    with socket.create_connection(("127.0.0.1", web_port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise TimeoutError("Rerun web server did not start within 10 seconds")
        except BaseException:
            self._stop_web()
            raise
        uri = f"rerun+http://127.0.0.1:{grpc_port}/proxy"
        print(f"[viewer] http://127.0.0.1:{web_port}/?url={quote(uri, safe='')} "
              "(available while this process runs)", flush=True)
        return uri


    def _stop_web(self):
        if self.web_process is not None:
            self.web_process.terminate()
            try:
                self.web_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.web_process.kill()
                self.web_process.wait(timeout=3)
            self.web_process = None


    def close(self):
        try:
            self.recording.flush(timeout_sec=5)
        finally:
            try:
                self.recording.disconnect()
            finally:
                self._stop_web()

