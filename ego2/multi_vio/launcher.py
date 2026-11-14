"""Native preparation panel; no camera streams or Rerun exist in this process."""
import argparse
import json
import os
from pathlib import Path
import sys

# The Conda OpenCV wheel and PySide6 bundle different GLib/Qt libraries.
# Resolve calibration/OpenCV first, then remove only OpenCV's Qt5 plugin
# overrides before initializing Qt6. Capture itself remains in another process.
# Load OpenCV/calibration before Qt, including when launched as a subprocess.
from . import config as _config
from .modes import MODES, HINTS, load_mode_config, validate_mode
from .selection import (LABELS, ROLES, default_selection, preferences_path,
    qt_environment, scan_devices, validate_connected, write_preferences, default_mode)
for _key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
    if _key not in qt_environment():
        os.environ.pop(_key, None)

from PySide6.QtCore import Qt, QPointF, QThread, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget)
from ..common.qt_runtime import configure_plugins
configure_plugins()

COLORS = dict(chest='#50a0ff', head='#c18aff', left_hand='#ffa546', right_hand='#3fdcbe')
STYLE = '''
QWidget { background:#202226; color:#edf0f4; font-family:"Noto Sans CJK SC"; font-size:15px; }
QLabel { background:transparent; border:none; }
QLabel#title { font-size:27px; font-weight:600; }
QLabel#sub { color:#a8afb9; font-size:14px; }
QLabel#section { font-size:16px; font-weight:600; }
QLabel#good { color:#6ed4aa; font-size:13px; }
QLabel#warning { color:#e8b56f; font-size:13px; }
QFrame#device { background:#282b30; border:1px solid #3a3e44; border-radius:8px; }
QFrame#line { background:#3a3e44; border:none; }
QScrollArea { border:none; background:transparent; }
QPushButton { background:#30343b; border:1px solid #484e58; border-radius:7px; padding:10px 19px; }
QPushButton:hover { background:#3b414a; }
QPushButton#start { background:#4285e9; border:1px solid #5997f4; color:white; font-weight:600; padding:12px 25px; }
QPushButton#start:hover { background:#5594f0; }
QPushButton:disabled, QPushButton#start:disabled { background:#2c333e; border-color:#39414c; color:#737d8b; }
QComboBox { background:#292d33; border:1px solid #4a515e; border-radius:7px; padding:10px 14px; min-height:23px; }
QComboBox:focus { border-color:#5997f4; }
QComboBox::drop-down { width:31px; border:none; }
QComboBox QAbstractItemView { background:#30343b; selection-background-color:#4285e9; }
QCheckBox { background:transparent; spacing:10px; color:#b9c0c9; font-size:14px; }
QCheckBox::indicator { width:18px; height:18px; }
'''


class CameraCombo(QComboBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor('#b4bdcb'), 1.7))
        x,y = self.width()-22, self.height()/2
        painter.drawPolyline([QPointF(x-5,y-2), QPointF(x,y+3), QPointF(x+5,y-2)])


class ScanThread(QThread):
    def __init__(self, scanner, parent):
        super().__init__(parent)
        self.scanner = scanner
        self.devices, self.error = [], ''

    def run(self):
        try:
            self.devices = self.scanner()
        except Exception as exc:
            self.error = f'设备检查失败：{exc}'


def label(text, name=None):
    widget = QLabel(text)
    widget.setTextFormat(Qt.PlainText)
    if name:
        widget.setObjectName(name)
    return widget


class PreparationDialog(QDialog):
    def __init__(self, config_path, *, scanner=None, preference_file=None,
                 initial_selection=None, auto_scan=True, mode=None):
        super().__init__()
        self.config_path = Path(config_path).expanduser().resolve()
        self.preference_file = Path(preference_file) if preference_file else preferences_path(config_path)
        self.scanner = scanner or (lambda: scan_devices(self.config_path, self.mode))
        self.devices = []
        self.worker = None
        self.selection = None
        self.start_pending = False
        self.close_pending = False
        self.first_scan = True
        self.notice = ''
        try:
            self.mode = validate_mode(mode) if mode is not None else default_mode(self.config_path)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.notice = str(exc)
            self.mode = 'vio'
        if initial_selection is None:
            try:
                # Restore a saved non-chest choice even when switching to VIO;
                # show its validation error and let the user amend the choice.
                initial_selection = default_selection(self.config_path)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.notice = str(exc)
                initial_selection = {}
        self.initial_selection = initial_selection
        self.setWindowTitle('多相机数据采集准备')
        self.setStyleSheet(STYLE)
        self.resize(1180, 720)
        self.setMinimumSize(1060, 660)
        self._build()
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh_devices)
        if auto_scan:
            self.timer.start()
            QTimer.singleShot(0, self.refresh_devices)
        self._update_validation()

    def _build(self, host=None):
        layout = QVBoxLayout(host if host is not None else self)
        layout.setContentsMargins(28,24,28,24)
        layout.setSpacing(18)
        header = QHBoxLayout()
        titles = QVBoxLayout(); titles.setSpacing(6)
        titles.addWidget(label('多相机数据采集准备', 'sub'))
        titles.addWidget(label('选择相机与佩戴部位对应关系', 'title'))
        titles.addWidget(label('选择完成后启动Rerun', 'sub'))
        header.addLayout(titles); header.addStretch()
        mode_box = QVBoxLayout()
        mode_box.addWidget(label('运行模式', 'sub'))
        self.mode_combo = CameraCombo(); self.mode_combo.setMinimumWidth(310)
        for key, title in MODES.items(): self.mode_combo.addItem(title, key)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(self.mode))
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        mode_box.addWidget(self.mode_combo)
        header.addLayout(mode_box)
        layout.addLayout(header)
        self._separator(layout)
        body = QHBoxLayout(); body.setSpacing(30)
        left = QVBoxLayout(); left.setSpacing(13)
        self.device_title = label('正在识别相机…', 'section')
        device_header = QHBoxLayout()
        device_header.addWidget(self.device_title); device_header.addStretch()
        self.refresh_button = QPushButton('刷新设备')
        self.refresh_button.setFixedHeight(44)
        self.refresh_button.clicked.connect(self.refresh_devices)
        device_header.addWidget(self.refresh_button)
        left.addLayout(device_header)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        self.cards = QWidget()
        self.cards_layout = QVBoxLayout(self.cards)
        self.cards_layout.setContentsMargins(0,0,6,0)
        self.cards_layout.setSpacing(12)
        scroll.setWidget(self.cards)
        left.addWidget(scroll, 1)
        body.addLayout(left, 4)
        right = QVBoxLayout(); right.setSpacing(12)
        right.addWidget(label('部位分配', 'section'))
        grid = QGridLayout(); grid.setHorizontalSpacing(22); grid.setVerticalSpacing(12)
        self.combos = {}
        self.part_labels = {}
        for row,role in enumerate(ROLES):
            name = LABELS[role]+('（多相机参考）' if role == 'chest' else '')
            part = QLabel(f'<span style="color:{COLORS[role]}">●</span>　{name}')
            self.part_labels[role] = part
            grid.addWidget(part, row, 0)
            combo = CameraCombo(); combo.setMinimumWidth(250)
            combo.setAccessibleName(name+'相机')
            combo.addItem('不启用', None)
            combo.currentIndexChanged.connect(self._selection_changed)
            self.combos[role] = combo
            grid.addWidget(combo, row, 1)
        grid.setColumnStretch(1, 1)
        right.addLayout(grid)
        self.policy = label('', 'sub')
        self.policy.setWordWrap(True)
        right.addWidget(self.policy)
        self.validation = label('', 'warning'); self.validation.setWordWrap(True)
        right.addWidget(self.validation)
        right.addStretch()
        body.addLayout(right, 6)
        layout.addLayout(body, 1)
        self._separator(layout)
        texts = QVBoxLayout(); texts.setSpacing(6)
        texts.addWidget(label('提示', 'section'))
        self.hint = label('')
        self.hint.setWordWrap(True); self.hint.setStyleSheet('color:#c4cad2;font-size:14px;')
        texts.addWidget(self.hint); layout.addLayout(texts)
        footer = QHBoxLayout(); footer.setSpacing(12)
        self.remember = QCheckBox('记住这次模式与分配，下次自动填入'); self.remember.setChecked(True)
        footer.addWidget(self.remember); footer.addStretch()
        self.cancel_button = QPushButton('取消'); self.cancel_button.clicked.connect(self.reject)
        footer.addWidget(self.cancel_button)
        self.start_button = QPushButton('启动 0 台相机'); self.start_button.setObjectName('start')
        self.start_button.clicked.connect(self.start)
        self.start_button.setAutoDefault(False)
        footer.addWidget(self.start_button)
        layout.addLayout(footer)
        self._mode_labels()

    def _mode_labels(self, *_):
        self.hint.setText(HINTS[self.mode])
        self.policy.setText('单相机可选任意部位；启用多相机须包含胸部。' if self.mode == 'vio' else
                            '三维重建仅支持一台相机，可选择任意部位。' if self.mode == 'reconstruction' else
                            '当前模式支持任意部位组合，无需选择胸部。')
        name = '胸部' + ('（多相机参考）' if self.mode == 'vio' else '')
        self.part_labels['chest'].setText(f'<span style="color:{COLORS["chest"]}">●</span>　{name}')
        self.combos['chest'].setAccessibleName(name+'相机')

    def _mode_changed(self, *_):
        self.mode = self.mode_combo.currentData()
        self.notice = ''
        self._mode_labels()
        self._update_validation()
        self.refresh_devices()

    @staticmethod
    def _separator(layout):
        line = QFrame(); line.setObjectName('line'); line.setFixedHeight(1)
        layout.addWidget(line)

    def current_selection(self):
        return {r:c.currentData() for r,c in self.combos.items() if c.currentData() is not None}

    def refresh_devices(self):
        if self.worker is not None or self.close_pending:
            return
        self.worker = ScanThread(self.scanner, self)
        self.mode_combo.setEnabled(False)
        self.worker.finished.connect(self._scan_finished)
        self.refresh_button.setEnabled(False)
        self.refresh_button.setText('正在检查…')
        self._update_validation()
        self.worker.start()

    def _scan_finished(self):
        worker, self.worker = self.worker, None
        self.mode_combo.setEnabled(True)
        worker.deleteLater()
        self.refresh_button.setEnabled(True)
        self.refresh_button.setText('刷新设备')
        if self.close_pending:
            super().reject()
            return
        if worker.error:
            self.notice = worker.error
            self.start_pending = False
            for combo in self.combos.values(): combo.setEnabled(True)
            self.apply_devices([])
            return
        self.apply_devices(worker.devices)
        if self.start_pending:
            self._finish_start()

    def apply_devices(self, devices):
        """Preserve edits across refreshes, including an unplugged selected SN."""
        if not self.first_scan and devices == self.devices:
            # Do not clear an open dropdown every two seconds when nothing
            # changed. Its model and keyboard/pointer interaction stay intact.
            self._update_validation()
            return
        selected = self.current_selection()
        if self.first_scan:
            ready = {d.sn for d in devices if d.ready}
            selected = {r:sn for r,sn in self.initial_selection.items() if r in ROLES and sn in ready}
            self.first_scan = False
        self.devices = devices
        for role,combo in self.combos.items():
            combo.blockSignals(True)
            combo.clear(); combo.addItem('不启用', None)
            seen = set()
            for device in devices:
                if not device.sn or device.sn in seen:
                    continue
                seen.add(device.sn)
                combo.addItem(device.sn+('（不可用）' if not device.ready else ''), device.sn)
                combo.setItemData(combo.count()-1, device.problem, Qt.ToolTipRole)
            sn = selected.get(role)
            if sn is not None and sn not in seen:
                combo.addItem(sn+'（已断开）', sn)
            combo.setCurrentIndex(max(0, combo.findData(sn)))
            combo.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self, *_):
        selected = self.current_selection()
        for role,combo in self.combos.items():
            used = {sn.lower() for r,sn in selected.items() if r != role}
            for index in range(1, combo.count()):
                sn = combo.itemData(index)
                matches = [d for d in self.devices if d.sn == sn]
                ready = len(matches) == 1 and matches[0].ready
                combo.model().item(index).setEnabled(ready and sn.lower() not in used)
        self._render_cards()
        self._update_validation()

    def _render_cards(self):
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.device_title.setText(f'已连接相机  ·  {len(self.devices)} 台')
        assigned = {sn:LABELS[r] for r,sn in self.current_selection().items()}
        if not self.devices:
            empty = label('未发现 Cyperstereo 相机。\n请连接 USB 后刷新设备。', 'sub')
            empty.setWordWrap(True); self.cards_layout.addWidget(empty)
        for device in self.devices:
            card = QFrame(); card.setObjectName('device')
            box = QVBoxLayout(card); box.setContentsMargins(16,13,16,13); box.setSpacing(5)
            row = QHBoxLayout()
            sn = label(device.sn or '未知 SN'); sn.setStyleSheet('font-size:19px;font-weight:600;')
            row.addWidget(sn); row.addStretch(); row.addWidget(label(device.model, 'sub'))
            box.addLayout(row)
            if device.ready:
                row = QHBoxLayout()
                row.addWidget(label('●  标定参数可用', 'good')); row.addStretch()
                row.addWidget(label('已分配：'+assigned[device.sn] if device.sn in assigned else '未分配', 'sub'))
                box.addLayout(row)
            else:
                problem = label('●  '+device.problem[:180], 'warning')
                problem.setWordWrap(True); problem.setToolTip(device.problem)
                box.addWidget(problem)
            card.setToolTip(f'{device.node}\n{device.usb_path}')
            self.cards_layout.addWidget(card)
        self.cards_layout.addStretch()

    def _update_validation(self):
        error = ''
        try:
            validate_connected(self.current_selection(), self.devices, mode=self.mode)
        except ValueError as exc:
            error = str(exc)
        self.validation.setText(self.notice or error)
        self.validation.setToolTip(self.notice or error)
        self.start_button.setText('正在检查…' if self.start_pending else f'启动 {len(self.current_selection())} 台相机')
        self.start_button.setEnabled(not error and self.worker is None and not self.start_pending)

    def start(self):
        if self.worker is not None:
            return
        try:
            validate_connected(self.current_selection(), self.devices, mode=self.mode)
        except ValueError:
            self._update_validation()
            return
        self.notice = ''
        self.start_pending = True
        for combo in self.combos.values(): combo.setEnabled(False)
        self.refresh_devices()  # Fresh USB and parameter check at acceptance.

    def _finish_start(self):
        self.start_pending = False
        try:
            selected = validate_connected(self.current_selection(), self.devices, mode=self.mode)
            # Validate the combined role mapping too (including custom row
            # order), before remembering it or handing control to capture.
            load_mode_config(self.config_path, mode=self.mode, selection=selected)
            if self.remember.isChecked():
                write_preferences(self.preference_file, selected, mode=self.mode)
            else:
                self.preference_file.unlink(missing_ok=True)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.notice = str(exc)
            for combo in self.combos.values(): combo.setEnabled(True)
            self._update_validation()
            return
        self.selection = selected
        self.timer.stop()
        self.accept()

    def reject(self):
        self.timer.stop()
        self.start_pending = False
        if self.worker is not None:
            self.close_pending = True
            self.setEnabled(False)
        else:
            super().reject()

    def closeEvent(self, event):
        event.ignore()
        self.reject()


def main(argv=None):
    parser = argparse.ArgumentParser(description='多相机数据采集准备')
    parser.add_argument('--config', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--mode', choices=tuple(MODES))
    args = parser.parse_args(argv)
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or
            os.environ.get('QT_QPA_PLATFORM') in ('offscreen','minimal')):
        print('[error] 未发现桌面显示环境。请在桌面终端启动，或使用 --no-setup。', file=sys.stderr)
        return 1
    app = QApplication([]); app.setStyle('Fusion'); app.setFont(QFont('Noto Sans CJK SC', 11))
    dialog = PreparationDialog(args.config, mode=args.mode)
    if dialog.exec() != QDialog.Accepted:
        return 2
    Path(args.result).write_text(json.dumps(dict(mode=dialog.mode, devices=dialog.selection), ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
