"""Visible save progress; closing this window never cancels the writer."""
import argparse
import json
from pathlib import Path
import time

from .launcher import QApplication, QDialog, QVBoxLayout, QPushButton, QTimer, STYLE, label
from PySide6.QtWidgets import QProgressBar


class SavingDialog(QDialog):
    def __init__(self, path, channel):
        super().__init__()
        self.path, self.channel = Path(path), Path(channel)
        self.setWindowTitle('Data Collector — 正在保存')
        self.setStyleSheet(STYLE)
        self.resize(640, 300)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        self.heading = label('正在保存', 'title')
        self.stage = label('等待采集缓存写入完成…', 'section')
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.elapsed = label('', 'sub')
        self.location = label(str(self.path), 'sub')
        self.location.setWordWrap(True)
        self.notice = label('关闭此窗口会继续后台保存；完成后显示保存成功面板。', 'sub')
        self.notice.setWordWrap(True)
        self.hide_button = QPushButton('后台保存')
        self.hide_button.clicked.connect(self.accept)
        for widget in (self.heading, self.stage, self.bar, self.elapsed, self.location, self.notice, self.hide_button):
            layout.addWidget(widget)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(200)
        self.poll()

    def poll(self):
        try:
            data = json.loads((self.channel / 'save-progress.json').read_text())
        except (OSError, ValueError):
            return
        if data.get('path') != str(self.path):
            return
        if data.get('phase') == 'complete':
            self.accept()
            return
        if data.get('phase') == 'error':
            self.heading.setText('保存未完成')
            self.stage.setText(str(data.get('stage', '请查看终端输出')))
            self.stage.setWordWrap(True)
            self.bar.hide()
            self.notice.setText('录制缓存已保留。此窗口关闭后不会删除数据。')
            self.hide_button.setText('关闭')
        else:
            current, total = int(data.get('current', 0)), int(data.get('total', 0))
            self.stage.setText(str(data.get('stage', '正在保存')))
            if total > 0:
                self.bar.setRange(0, 1000)
                self.bar.setValue(min(1000, int(current * 1000 / total)))
                self.bar.setFormat(f'{current} / {total}（当前步骤）')
            else:
                self.bar.setRange(0, 0)
        seconds = max(0, int(time.monotonic() - data.get('started', time.monotonic())))
        self.elapsed.setText(f'已等待 {seconds // 60} 分 {seconds % 60} 秒')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--channel', required=True)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    dialog = SavingDialog(args.path, args.channel)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    dialog.exec()


if __name__ == '__main__':
    main()
