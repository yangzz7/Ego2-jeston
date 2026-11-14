"""Exercise the actual Qt dialog without cameras or a visible desktop."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
from pathlib import Path
import tempfile
import unittest
import cv2
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
from ego2.multi_vio.record_panel import RecordingDialog,QApplication,QDialog
from ego2.multi_vio.storage import write_json


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def test_cancel_and_enter_do_not_start_recording(self):
        with tempfile.TemporaryDirectory() as d:
            panel=RecordingDialog('gesture',d);panel.show();self.app.processEvents()
            QTest.keyClick(panel.directory,Qt.Key_Return)
            self.assertFalse((Path(d)/'request.json').exists())
            QTest.mouseClick(panel.cancel,Qt.LeftButton)
            self.assertEqual(panel.result(),QDialog.Rejected)
            self.assertFalse(list(Path(d).iterdir()))

    def test_submit_waits_for_ack_failure_can_be_fixed_success_closes(self):
        with tempfile.TemporaryDirectory() as d:
            channel=Path(d);panel=RecordingDialog('aprilgrid',channel)
            panel.directory.setText(d);panel.name.setText('试验01');panel.show();self.app.processEvents()
            self.assertIn('/aprilgrid/试验01',panel.preview.text())
            QTest.mouseClick(panel.start,Qt.LeftButton)
            request=json.loads((channel/'request.json').read_text())
            self.assertTrue(panel.pending);self.assertFalse((channel/'aprilgrid').exists())
            write_json(channel/'response.json',dict(id=request['id'],ok=False,error='保存位置不可写'))
            panel.poll();self.assertFalse(panel.pending);self.assertTrue(panel.isVisible())
            self.assertIn('不可写',panel.notice.text())
            panel.submit();request=json.loads((channel/'request.json').read_text())
            write_json(channel/'response.json',dict(id=request['id'],ok=True))
            panel.poll();self.assertEqual(panel.result(),QDialog.Accepted);self.assertFalse(panel.isVisible())


if __name__=='__main__':unittest.main()
