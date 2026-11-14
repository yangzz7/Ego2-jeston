"""Focus isolation and shortcut-to-dialog integration without camera hardware."""
from io import StringIO
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ego2.multi_vio import manual
from ego2.multi_vio.record_shortcut import ViewerShortcut, X11Enter, belongs_to
from ego2.common.qt_runtime import loaded_plugin_directory


class ShortcutTests(unittest.TestCase):
    def test_qt_plugins_follow_loaded_conda_library_only_if_available(self):
        with tempfile.TemporaryDirectory() as directory:
            lib = Path(directory)/'lib'
            platform = lib/'qt6/plugins/platforms/libqxcb.so'
            maps = f'001-002 r-xp 0 00:00 1 {lib}/libQt6Core.so.6.11.1'
            self.assertIsNone(loaded_plugin_directory(maps))
            platform.parent.mkdir(parents=True); platform.touch()
            self.assertEqual(loaded_plugin_directory(maps), lib/'qt6/plugins')
            self.assertIsNone(loaded_plugin_directory(maps.replace('libQt6Core','libQt5Core')))

    def listener(self):
        listener = X11Enter.__new__(X11Enter)
        listener.owner = os.getpid()
        listener.allow_external = False
        listener.root = 1
        listener.keys = {36, 104}
        listener.unmodified = Mock(return_value=True)
        listener.property = Mock(side_effect=lambda window, name: {
            '_NET_ACTIVE_WINDOW': 123, 'WM_CLASS': b'\0rerun\0',
            '_NET_WM_PID': os.getpid()}[name])
        return listener

    def test_only_enter_on_own_foreground_rerun_without_modifiers_or_repeat(self):
        listener = self.listener()
        for key in (36, 104):
            self.assertTrue(listener.triggers(key, 0))
        self.assertFalse(listener.triggers(38, 0))
        self.assertFalse(listener.triggers(36, 1 << 16))
        listener.unmodified.return_value = False
        self.assertFalse(listener.triggers(36, 0))
        listener.unmodified.return_value = True
        for properties in ((123, b'firefox\0Firefox\0'), (123, b'record_panel\0'), (None,)):
            listener.property = Mock(side_effect=properties)
            self.assertFalse(listener.triggers(36, 0))
        listener.property = Mock(side_effect=(123, b'\0rerun\0', 1))
        self.assertFalse(listener.triggers(36, 0))

    def test_process_descendants_match_but_unrelated_viewer_does_not(self):
        child = subprocess.Popen(['sleep', '10'])
        try:
            self.assertTrue(belongs_to(child.pid, os.getpid()))
            self.assertFalse(belongs_to(os.getpid(), child.pid))
            self.assertFalse(belongs_to(999999999, os.getpid()))
        finally:
            child.terminate(); child.wait()

    @patch.dict(os.environ, {}, clear=True)
    def test_no_display_does_not_spawn_and_mailbox_is_consumed_once(self):
        with tempfile.TemporaryDirectory() as directory, patch('subprocess.Popen') as popen:
            shortcut = ViewerShortcut(directory, SimpleNamespace())
            popen.assert_not_called()
            path = Path(directory)/'open-panel'
            path.touch()
            self.assertTrue(shortcut.poll())
            self.assertFalse(shortcut.poll())
            shortcut.close()

    @patch.dict(os.environ, {'DISPLAY': ':1'}, clear=True)
    def test_helper_start_and_shutdown_and_web_exclusion(self):
        with tempfile.TemporaryDirectory() as directory, patch('subprocess.Popen') as popen:
            process = popen.return_value
            process.poll.return_value = None
            shortcut = ViewerShortcut(directory, SimpleNamespace())
            self.assertIn('ego2.multi_vio.record_shortcut', popen.call_args.args[0])
            shortcut.close()
            process.terminate.assert_called_once(); process.wait.assert_called_once()
            popen.reset_mock()
            ViewerShortcut(directory, SimpleNamespace(web=True)).close()
            popen.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_shortcut_opens_and_raises_dialog_without_creating_dataset(self):
        with patch('subprocess.Popen') as popen:
            popen.return_value.poll.return_value = None
            prepare = Mock()
            c = manual.ManualController('gesture', SimpleNamespace(session=None),
                prepare, Mock(), Mock(), Mock(), stdin=StringIO(''))
            try:
                (c.directory/'open-panel').touch(); c.poll()
                popen.assert_called_once()
                prepare.assert_not_called(); self.assertIsNone(c.root)
                (c.directory/'open-panel').touch(); c.poll()
                popen.assert_called_once()
                self.assertTrue((c.directory/'raise-panel').exists())
            finally:
                c.close()


if __name__ == '__main__':
    unittest.main()
