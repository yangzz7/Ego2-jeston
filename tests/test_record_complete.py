"""Completion notifications follow successful saving without blocking preview."""
from concurrent.futures import Future
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from ego2.multi_vio.record_complete import RecordingCompleteDialog, QApplication, Qt, QUrl
from PySide6.QtTest import QTest
from ego2.multi_vio.manual import ManualController
from ego2.multi_vio.storage import status


class CompletionControlTests(unittest.TestCase):
    def setUp(self):
        shortcut = self.enterContext(patch('ego2.multi_vio.manual.ViewerShortcut')).return_value
        shortcut.poll.return_value = False
        self.enterContext(patch.dict(os.environ, DISPLAY=':test'))
        self.popen = self.enterContext(patch('ego2.multi_vio.manual.subprocess.Popen'))
        self.popen.return_value.poll.return_value = None
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def controller(self, mode='gesture'):
        controller = ManualController(mode, SimpleNamespace(session=None, headless=False),
            Mock(), Mock(), Mock(), Mock(), stdin=StringIO(''))
        self.addCleanup(controller.close)
        controller.root = self.directory/mode
        controller.root.mkdir()
        controller.active = controller.stopping = True
        controller.finish_future = Future()
        status(controller.root, 'finalizing')
        return controller

    def test_all_modes_notify_only_after_success_once_with_exact_path(self):
        for mode in ('vio', 'gesture', 'aprilgrid', 'reconstruction'):
            with self.subTest(mode=mode):
                c = self.controller(mode)
                self.popen.reset_mock()
                c.wait_finalization(block=False, notify=True)
                self.popen.assert_not_called()
                status(c.root, 'complete')
                c.finish_future.set_result(None)
                c.wait_finalization(block=False, notify=True)
                c.wait_finalization(block=False, notify=True)
                self.popen.assert_called_once()
                command = self.popen.call_args.args[0]
                self.assertIn('ego2.multi_vio.record_complete', command)
                self.assertEqual(command[command.index('--path')+1], str(c.root))
                self.assertEqual(c.saved, [c.root])
                self.assertFalse(c.active or c.stopping)

    def test_save_error_has_no_success_notification(self):
        c = self.controller()
        c.finish_future.set_exception(RuntimeError('disk full'))
        c.wait_finalization(notify=True)
        self.popen.assert_not_called()
        self.assertFalse(c.saved)
        self.assertEqual(json.loads((c.root/'session.json').read_text())['status'], 'incomplete')

    def test_popup_launch_error_does_not_invalidate_dataset(self):
        c = self.controller()
        status(c.root, 'complete')
        self.popen.side_effect = OSError('cannot launch Qt')
        c.finish_future.set_result(None)
        c.wait_finalization(notify=True)
        self.assertEqual(c.saved, [c.root])
        self.assertEqual(json.loads((c.root/'session.json').read_text())['status'], 'complete')
        self.assertIsNone(c.completion_process)

    def test_existing_completion_raised_then_next_enter_opens_settings_and_cleanup(self):
        c = self.controller()
        c.finish_future.set_result(None)
        c.wait_finalization(notify=True)
        process = c.completion_process
        c.open()
        self.assertTrue((c.directory/'raise-completion').exists())
        self.popen.assert_called_once()
        process.poll.return_value = 0
        process.returncode = 0
        c.poll()
        self.assertIsNone(c.completion_process)
        next_process = Mock()
        next_process.poll.return_value = None
        self.popen.return_value = next_process
        c.open()
        self.assertIn('ego2.multi_vio.record_panel', self.popen.call_args.args[0])
        c.close()
        next_process.terminate.assert_called_once()
        next_process.poll.return_value = 0

    def test_shutdown_and_headless_do_not_open_preview_notification(self):
        c = self.controller()
        c.finish_future.set_result(None)
        c.wait_finalization()
        self.popen.assert_not_called()
        c.args.headless = True
        c.show_completion(c.root)
        self.popen.assert_not_called()
        c.args.headless = False
        with patch.dict(os.environ, DISPLAY='', WAYLAND_DISPLAY=''):
            c.show_completion(c.root)
        self.popen.assert_not_called()


class CompletionDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_path_is_plain_selectable_and_folder_button_opens_exact_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'中文 & <b> 数据集'
            path.mkdir()
            dialog = RecordingCompleteDialog(path)
            dialog.show()
            self.app.processEvents()
            self.assertEqual(dialog.location.toPlainText(), str(path))
            self.assertTrue(dialog.location.isReadOnly())
            with patch('ego2.multi_vio.record_complete.QDesktopServices.openUrl', return_value=True) as open_url:
                QTest.mouseClick(dialog.open_folder, Qt.LeftButton)
                open_url.assert_called_once_with(QUrl.fromLocalFile(str(path)))
            self.assertTrue(dialog.isVisible())
            QTest.mouseClick(dialog.confirm, Qt.LeftButton)
            self.assertFalse(dialog.isVisible())

    def test_open_error_is_visible_and_enter_or_close_only_dismisses(self):
        with tempfile.TemporaryDirectory() as directory:
            dialog = RecordingCompleteDialog(directory)
            dialog.show()
            with patch('ego2.multi_vio.record_complete.QDesktopServices.openUrl', return_value=False):
                dialog.show_folder()
                self.assertTrue(dialog.notice.isVisible())
                self.assertIn('无法打开', dialog.notice.text())
            with patch('ego2.multi_vio.record_complete.QDesktopServices.openUrl') as open_url:
                dialog.confirm.setFocus()
                QTest.keyClick(dialog.confirm, Qt.Key_Return)
                self.assertFalse(dialog.isVisible())
                open_url.assert_not_called()
                dialog.show()
                dialog.close()
                self.assertFalse(dialog.isVisible())


if __name__ == '__main__':
    unittest.main()
