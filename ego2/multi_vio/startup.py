"""Sensia's capture/replay tabs; camera setup is kept independent of playback."""
import argparse
import json
import os
from pathlib import Path
import sys

from .launcher import (PreparationDialog, QApplication, QDialog, QVBoxLayout,
                       QWidget, QTimer, QFont, STYLE)
from PySide6.QtWidgets import QTabWidget
from .modes import MODES
from .replay_panel import ReplayPage


class StartupDialog(PreparationDialog):
    def __init__(self, config_path, *, output=None, auto_scan=True, **kwargs):
        self.output = output
        self.auto_scan = auto_scan
        self.replay_ready = False
        super().__init__(config_path, auto_scan=False, **kwargs)
        self.setWindowTitle('Sensia')
        self.resize(1220, 830)
        self.setMinimumSize(1100, 770)
        self.setStyleSheet(STYLE + '''
QTabWidget::pane { border:none; }
QTabBar::tab { background:#25292f; color:#b6bfcb; padding:13px 28px; border-bottom:3px solid transparent; }
QTabBar::tab:selected { color:#65a1ff; background:#293342; border-bottom:3px solid #4285e9; }
QLineEdit { background:#292d33; border:1px solid #4a515e; border-radius:7px; padding:12px; }
QListWidget { background:transparent; border:none; outline:none; }
QListWidget::item { background:#282c32; border:1px solid #424953; border-radius:7px; padding:12px; }
QListWidget::item:selected { background:#2c3c54; border:1px solid #5997f4; }
''')
        self.tabs.currentChanged.connect(self.tab_changed)
        self.replay.cancel.clicked.connect(self.cancel_replay)
        self.replay.launched.connect(self.replay_launched)
        self.replay.busyChanged.connect(lambda busy: self.tabs.tabBar().setEnabled(not busy))
        self.tab_changed(0)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.live_page = QWidget()
        self.tabs.addTab(self.live_page, '实时采集')
        super()._build(self.live_page)
        self.replay = ReplayPage(output=self.output) if self.output is not None else ReplayPage()
        self.tabs.addTab(self.replay, '数据回放')
        layout.addWidget(self.tabs)

    def tab_changed(self, index):
        if index == 0:
            if self.auto_scan:
                self.timer.start()
                self.refresh_devices()
        else:
            self.timer.stop()
            if not self.replay.scanned: self.replay.scan()

    def refresh_devices(self):
        if self.tabs.currentIndex() == 0: super().refresh_devices()

    def start(self):
        super().start()
        if self.start_pending: self.tabs.tabBar().setEnabled(False)

    def _scan_finished(self):
        super()._scan_finished()
        if not self.start_pending and not self.replay.busy:
            self.tabs.tabBar().setEnabled(True)
        if self.replay_ready: self.replay_launched()

    def replay_launched(self):
        self.replay_ready = True
        self.timer.stop()
        if self.worker is None: self.accept()

    def cancel_replay(self):
        if self.replay.busy:
            # A final camera scan may still be finishing after the Viewer
            # starts. Cancellation must revoke that deferred acceptance too.
            self.replay_ready = False
            self.replay.cancel_preparation()
        else: self.reject()

    def reject(self):
        if self.replay.busy:
            self.replay.cancel_preparation()
            self.replay_ready = False
            return
        super().reject()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sensia 实时采集与数据回放')
    parser.add_argument('--config', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--mode', choices=tuple(MODES))
    args = parser.parse_args(argv)
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or
            os.environ.get('QT_QPA_PLATFORM') in ('offscreen', 'minimal')):
        print('[error] 未发现桌面显示环境。请在桌面终端启动，或使用 --no-setup。', file=sys.stderr)
        return 1
    app = QApplication([])
    app.setStyle('Fusion'); app.setFont(QFont('Noto Sans CJK SC', 11))
    dialog = StartupDialog(args.config, mode=args.mode)
    try:
        if dialog.exec() != QDialog.Accepted: return 2
        if dialog.replay_ready:
            # The worker owns the Viewer. Keep its temporary RRD available
            # until the Viewer exits, even though the setup panel is closed.
            code = dialog.replay.process.wait()
            if code: print('[replay] Rerun 异常退出，请查看终端日志。', file=sys.stderr)
            result = dict(action='replay', exit_code=code)
        else:
            result = dict(mode=dialog.mode, devices=dialog.selection)
        Path(args.result).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
        return 0
    finally:
        dialog.replay.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
