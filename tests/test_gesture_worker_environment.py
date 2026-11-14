import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ego2.gesture.worker_environment import prepare_worker_environment, BackendReporter, display_backend_status


class WorkerEnvironment(unittest.TestCase):
    def test_jetson_gpu_isolates_display_but_keeps_other_settings(self):
        env = dict(DISPLAY='localhost:10.0', WAYLAND_DISPLAY='wayland-0',
                   QT_QPA_PLATFORM='xcb', XAUTHORITY='/tmp/test-authority',
                   EGO2_GESTURE_BACKEND='auto', PATH='/usr/bin')
        with patch.dict(os.environ, env, clear=True), patch('pathlib.Path.is_file', return_value=True):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prepare_worker_environment({'backend':'cpu'}), ('DISPLAY','WAYLAND_DISPLAY'))
            self.assertEqual(dict(os.environ), {k:v for k,v in env.items() if k not in ('DISPLAY','WAYLAND_DISPLAY')})

    def test_explicit_cpu_and_non_jetson_keep_display(self):
        for jetson, backend in ((True,'cpu'), (False,'auto'), (False,'gpu')):
            with self.subTest(jetson=jetson, backend=backend):
                with patch.dict(os.environ, {'DISPLAY':'localhost:10.0'}, clear=True), patch('pathlib.Path.is_file', return_value=jetson):
                    self.assertEqual(prepare_worker_environment({'backend':backend}), ())
                    self.assertEqual(os.environ['DISPLAY'],'localhost:10.0')

    def test_backend_changes_are_atomic_and_visible_with_reason(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'status.json'
            reporter=BackendReporter(path)
            eyes=[NS(backend='gpu',requested_backend='auto',fallbacks=0) for _ in range(2)]
            reporter(eyes)
            old_time=path.stat().st_mtime_ns
            reporter(eyes)
            self.assertEqual(path.stat().st_mtime_ns,old_time)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                previous=display_backend_status(path,'head',None)
                display_backend_status(path,'head',previous)
            self.assertEqual(output.getvalue().count('[gesture-inference]'),1)
            self.assertIn('eye0=GPU',output.getvalue())
            eyes[1].backend='cpu';eyes[1].fallbacks=1;eyes[1].last_fallback='GPU inference failed: lost context'
            reporter(eyes)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                current=display_backend_status(path,'head',previous)
            self.assertEqual(current[1]['backend'],'cpu')
            self.assertIn('eye1=CPU',output.getvalue())
            self.assertIn('lost context',output.getvalue())
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_diagnostic_failure_does_not_interrupt_inference(self):
        reporter=BackendReporter('/unavailable/status.json')
        with patch('pathlib.Path.write_text', side_effect=OSError('read only')), contextlib.redirect_stdout(io.StringIO()):
            reporter([NS(backend='gpu',requested_backend='gpu',fallbacks=0)])
        self.assertEqual(reporter.previous[0]['backend'],'gpu')

    def test_missing_status_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(display_backend_status(Path(root)/'missing','head',None))


if __name__=='__main__':unittest.main()
