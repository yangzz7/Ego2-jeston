"""Recording dialog in its own Qt process; never owns or pauses a camera."""
import argparse
import json
from pathlib import Path
import time

from .launcher import QApplication, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, Qt, QTimer, STYLE, label
from PySide6.QtWidgets import QFileDialog, QLineEdit
from ..common.paths import OUTPUT
from .manual import destination
from .modes import MODES
from .storage import FOLDERS, write_json


class RecordingDialog(QDialog):
    def __init__(self, mode, channel, preset=None):
        super().__init__()
        self.mode, self.channel = mode, Path(channel)
        self.pending = False
        self.token = None
        self.setWindowTitle('Data Collector')
        self.setStyleSheet(STYLE + '''
QLineEdit { background:#292d33; border:1px solid #566171; border-radius:7px; padding:12px; }
QLineEdit:focus { border-color:#5997f4; }
QLabel#preview { background:#25292e; border:1px solid #414750; border-radius:7px; padding:16px; font-size:14px; }
''')
        self.resize(930, 650)
        layout = QVBoxLayout(self); layout.setContentsMargins(30,25,30,25); layout.setSpacing(16)
        layout.addWidget(label('准备开始录制数据', 'title'))
        layout.addWidget(label('设置保存位置和名称，点击开始录制后面板自动关闭。', 'sub'))
        fmt = 'LeRobot V3' if mode in ('vio','gesture') else 'Rerun 回放包'
        layout.addWidget(label(f'当前模式：{MODES[mode]}     ·     {fmt}', 'section'))
        layout.addWidget(label('预览运行中 · 尚未录制', 'good'))
        layout.addSpacing(5)
        layout.addWidget(label('存储根目录（选填）', 'section'))
        row = QHBoxLayout()
        self.directory = QLineEdit(); self.directory.setPlaceholderText(f'留空使用 {OUTPUT}')
        self.browse = QPushButton('浏览…'); self.browse.clicked.connect(self.choose_directory)
        row.addWidget(self.directory,1); row.addWidget(self.browse); layout.addLayout(row)
        layout.addWidget(label(f'自动按当前模式保存到 {FOLDERS[mode]} 子目录', 'sub'))
        layout.addWidget(label('数据集文件夹名称（选填）', 'section'))
        self.name = QLineEdit(); self.name.setPlaceholderText('留空使用实际创建时间')
        layout.addWidget(self.name)
        if preset:
            path = Path(preset).expanduser().resolve()
            self.directory.setText(str(path.parent.parent if path.parent.name == FOLDERS[mode] else path.parent))
            self.name.setText(path.name)
        self.preview = label('', 'preview'); self.preview.setWordWrap(True)
        self.preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.preview)
        self.notice = label('', 'warning'); self.notice.setWordWrap(True); layout.addWidget(self.notice)
        layout.addStretch()
        buttons = QHBoxLayout()
        buttons.addWidget(label('录制后再按 Enter 停止并保存；取消后继续预览。', 'sub'),1)
        self.cancel = QPushButton('取消'); self.cancel.clicked.connect(self.reject)
        self.start = QPushButton('开始录制'); self.start.setObjectName('start'); self.start.clicked.connect(self.submit)
        for button in (self.cancel,self.start,self.browse): button.setAutoDefault(False); button.setDefault(False)
        buttons.addWidget(self.cancel); buttons.addWidget(self.start); layout.addLayout(buttons)
        self.directory.textChanged.connect(self.update_preview); self.name.textChanged.connect(self.update_preview)
        self.timer = QTimer(self); self.timer.timeout.connect(self.poll); self.timer.start(100)
        self.update_preview()

    def choose_directory(self):
        path = QFileDialog.getExistingDirectory(self, '选择存储根目录', self.directory.text() or str(OUTPUT))
        if path: self.directory.setText(path)

    def update_preview(self):
        try:
            target = destination(self.mode,self.directory.text(),self.name.text())
            self.preview.setText('最终保存位置（预览）\n'+str(target))
            self.start.setEnabled(not self.pending)
        except ValueError as exc:
            self.preview.setText(str(exc)); self.start.setEnabled(False)

    def submit(self):
        if self.pending: return
        self.token = str(time.time_ns()); self.pending = True
        for widget in (self.directory,self.name,self.browse,self.start,self.cancel): widget.setEnabled(False)
        self.notice.setText('正在检查保存位置并准备各相机写入器，预览继续运行…')
        write_json(self.channel/'request.json',dict(id=self.token,directory=self.directory.text(),name=self.name.text()))

    def poll(self):
        raise_request = self.channel/'raise-panel'
        if raise_request.exists():
            raise_request.unlink(missing_ok=True)
            self.showNormal(); self.raise_(); self.activateWindow()
        if not self.pending: return
        path = self.channel/'response.json'
        if not path.exists(): return
        reply = json.loads(path.read_text())
        if reply.get('id') != self.token: return
        self.pending = False
        if reply.get('ok'):
            self.accept()
        else:
            for widget in (self.directory,self.name,self.browse,self.start,self.cancel): widget.setEnabled(True)
            self.notice.setText(reply.get('error','无法开始录制'))

    def reject(self):
        if not self.pending: super().reject()

    def closeEvent(self,event):
        if self.pending: event.ignore()
        else: super().closeEvent(event)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=tuple(MODES))
    parser.add_argument('--channel',required=True);parser.add_argument('--preset')
    args=parser.parse_args();app=QApplication.instance() or QApplication([])
    dialog=RecordingDialog(args.mode,args.channel,args.preset)
    QTimer.singleShot(0, lambda: (dialog.raise_(), dialog.activateWindow(), dialog.directory.setFocus()))
    return 0 if dialog.exec()==QDialog.Accepted else 2


if __name__=='__main__': raise SystemExit(main())
