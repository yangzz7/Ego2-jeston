"""Saved-session browser and nonblocking playback preparation for Sensia."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

# Import the existing Qt runtime setup before loading any Qt widgets.
from .launcher import label, QFrame, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QWidget, Qt, QTimer
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QLineEdit, QListWidget, QListWidgetItem
from ..common.paths import OUTPUT
from .roles import LABELS
from .selection import qt_environment
from .session_catalog import inspect_session, list_sessions, duration_text, CONTENTS


class ReplayPage(QWidget):
    launched = Signal()
    busyChanged = Signal(bool)

    def __init__(self, parent=None, *, output=OUTPUT):
        super().__init__(parent)
        self.output = Path(output)
        self.selected = None
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='ego2-session-catalog')
        self.jobs = []
        self.generation = 0
        self.process = None
        self.temporary = None
        self.cancel_time = None
        self.playing = False
        self.closing = False
        self.scanned = False
        self._build()
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(250)
        self.debounce.timeout.connect(self.inspect)
        self.path.textChanged.connect(self.path_changed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 24)
        layout.setSpacing(16)
        layout.addWidget(label('选择要回放的数据', 'title'))
        layout.addWidget(label('选择已保存的数据，启动 Rerun 回放', 'sub'))
        row = QHBoxLayout()
        row.addWidget(label('数据集文件夹', 'section'))
        self.path = QLineEdit()
        self.path.setAccessibleName('数据集文件夹')
        self.path.setPlaceholderText('选择或粘贴包含 session.json 的数据集文件夹路径')
        row.addWidget(self.path, 1)
        self.browse = QPushButton('浏览…')
        self.browse.clicked.connect(self.choose)
        row.addWidget(self.browse)
        layout.addLayout(row)
        body = QHBoxLayout()
        body.setSpacing(18)
        left = QVBoxLayout()
        header = QHBoxLayout()
        header.addWidget(label('默认目录中的数据', 'section'), 1)
        self.refresh = QPushButton('刷新')
        self.refresh.clicked.connect(self.scan)
        header.addWidget(self.refresh)
        left.addLayout(header)
        self.list = QListWidget()
        self.list.setAccessibleName('已保存的数据')
        self.list.setSpacing(5)
        self.list.itemSelectionChanged.connect(self.choose_list_item)
        left.addWidget(self.list, 1)
        body.addLayout(left, 6)
        card = QFrame()
        card.setObjectName('device')
        details = QVBoxLayout(card)
        details.setContentsMargins(20, 18, 20, 18)
        details.setSpacing(14)
        details.addWidget(label('数据详情', 'section'))
        grid = QGridLayout()
        grid.setVerticalSpacing(15)
        grid.setHorizontalSpacing(24)
        self.values = {}
        for index, (key, title) in enumerate((('mode','功能模式'), ('format','数据格式'),
                ('state','保存状态'), ('cameras','相机数量'), ('duration','录制时长'))):
            grid.addWidget(label(title, 'sub'), index, 0)
            value = label('—')
            value.setWordWrap(True)
            self.values[key] = value
            grid.addWidget(value, index, 1)
        grid.setColumnStretch(1, 1)
        details.addLayout(grid)
        details.addWidget(label('录制时的相机分配', 'sub'))
        self.devices = label('请选择一份数据')
        self.devices.setWordWrap(True)
        self.devices.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details.addWidget(self.devices)
        details.addStretch()
        self.content = label('', 'sub')
        self.content.setWordWrap(True)
        details.addWidget(self.content)
        body.addWidget(card, 5)
        layout.addLayout(body, 1)
        self.notice = label('请选择一份已保存的数据。', 'warning')
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        line = QFrame(); line.setObjectName('line'); line.setFixedHeight(1)
        layout.addWidget(line)
        layout.addWidget(label('提示', 'section'))
        hint = label('无需连接相机，完全重现对应数据录制时的情况', 'sub')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QHBoxLayout()
        buttons.addWidget(label('自动识别数据模式', 'sub'), 1)
        self.cancel = QPushButton('取消')
        self.start = QPushButton('启动回放')
        self.start.setObjectName('start')
        self.start.setEnabled(False)
        self.start.clicked.connect(self.start_replay)
        for button in (self.browse, self.refresh, self.cancel, self.start):
            button.setAutoDefault(False)
        buttons.addWidget(self.cancel); buttons.addWidget(self.start)
        layout.addLayout(buttons)

    @property
    def busy(self):
        return self.process is not None

    def scan(self):
        if self.busy or self.closing or any(kind == 'list' for kind, _, _ in self.jobs): return
        self.scanned = True
        self.refresh.setEnabled(False)
        self.refresh.setText('正在读取…')
        self.jobs.append(('list', None, self.pool.submit(list_sessions, self.output)))

    def choose(self):
        directory = str(self.selected.path if self.selected else self.output)
        path = QFileDialog.getExistingDirectory(self, '选择数据集文件夹', directory)
        if path: self.path.setText(path)

    def choose_list_item(self):
        item = self.list.currentItem()
        if item and item.data(Qt.UserRole): self.path.setText(item.data(Qt.UserRole))

    def path_changed(self):
        self.generation += 1
        self.selected = None
        self.start.setEnabled(False)
        self.path.setToolTip(self.path.text())
        self.render(None)
        self.notice.setText('正在读取数据详情…' if self.path.text().strip() else '请选择一份已保存的数据。')
        self.debounce.start()

    def inspect(self):
        if self.busy or self.closing or not self.path.text().strip(): return
        self.jobs.append(('detail', self.generation, self.pool.submit(inspect_session, self.path.text().strip())))

    def render(self, item):
        values = dict(mode=item.title, format=item.format or '未记录', state=item.state_text,
                      cameras=f'{len(item.devices)} 台' if item.devices else '未记录',
                      duration=duration_text(item.duration)) if item else {}
        for key, widget in self.values.items(): widget.setText(values.get(key, '—'))
        self.values['state'].setStyleSheet('color:#6ed4aa;' if item and item.playable else 'color:#e8b56f;')
        self.devices.setText('\n'.join(f'{LABELS.get(role, role)}    {sn}' for role, sn in item.devices.items())
                             if item and item.devices else '未记录' if item else '请选择一份数据')
        self.content.setText('回放内容：'+CONTENTS[item.mode] if item and item.mode in CONTENTS else '')

    def _render_list(self, items):
        self.list.blockSignals(True)
        self.list.clear()
        for session in items:
            item = QListWidgetItem(f'{session.path.name}\n{session.state_text}')
            item.setData(Qt.UserRole, str(session.path))
            item.setToolTip(str(session.path)+ ('\n'+session.error if session.error else ''))
            self.list.addItem(item)
            if str(session.path) == self.path.text(): self.list.setCurrentItem(item)
        if not items:
            item = QListWidgetItem('暂无已保存的数据\n可通过“浏览…”选择其他目录。')
            item.setFlags(Qt.NoItemFlags)
            self.list.addItem(item)
        self.list.blockSignals(False)

    def set_busy(self, busy):
        for widget in (self.path, self.browse, self.refresh, self.list): widget.setEnabled(not busy)
        self.start.setEnabled(not busy and self.selected is not None and self.selected.playable)
        self.start.setText('正在准备回放…' if busy else '启动回放')
        self.cancel.setText('取消准备' if busy else '取消')
        self.cancel.setEnabled(True)
        self.busyChanged.emit(busy)

    def start_replay(self):
        if self.busy or not self.selected or not self.selected.playable: return
        self.temporary = tempfile.TemporaryDirectory(prefix='ego2-panel-replay-')
        self.playing = False
        self.cancel_time = None
        self.set_busy(True)
        self.notice.setText('正在准备回放…')
        try:
            self.process = subprocess.Popen([sys.executable, '-m', 'ego2.multi_vio.replay_worker',
                '--session', str(self.selected.path), '--workspace', self.temporary.name],
                env=qt_environment(), start_new_session=True)
        except OSError as exc:
            self.cleanup()
            self.set_busy(False)
            self.notice.setText(f'无法准备回放：{exc}')

    def cancel_preparation(self):
        if not self.busy or self.cancel_time is not None: return
        self.cancel_time = time.monotonic()
        self.notice.setText('正在取消回放准备…')
        self.cancel.setEnabled(False)
        self.signal_process(signal.SIGTERM)

    def signal_process(self, sig):
        if self.process:
            try: os.killpg(self.process.pid, sig)
            except ProcessLookupError: pass

    def poll(self):
        for job in list(self.jobs):
            kind, generation, future = job
            if not future.done(): continue
            self.jobs.remove(job)
            if self.closing: continue
            try:
                result = future.result()
                if kind == 'list': self._render_list(result)
                elif generation == self.generation and not self.busy:
                    self.selected = result
                    self.render(result)
                    self.start.setEnabled(result.playable)
                    self.notice.setText(result.error)
            except Exception as exc:
                if kind == 'list' or generation == self.generation:
                    self.notice.setText(f'无法读取数据：{exc}')
            if kind == 'list':
                self.refresh.setText('刷新')
                self.refresh.setEnabled(not self.busy)
        if not self.process: return
        if self.cancel_time is not None:
            if time.monotonic()-self.cancel_time > 2: self.signal_process(signal.SIGKILL)
            if self.process.poll() is not None:
                self.signal_process(signal.SIGKILL)  # Also stop any remaining decoder children.
                self.cleanup()
                self.set_busy(False)
                self.notice.setText('已取消回放准备，原始数据未修改。')
            return
        try:
            path = Path(self.temporary.name)/'state.json'
            state = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, ValueError): state = {}
        if state.get('phase') == 'playing' and not self.playing:
            self.playing = True
            self.launched.emit()
            return
        if self.process.poll() is not None and not self.playing:
            self.notice.setText(state.get('error') or f'回放准备失败（退出码 {self.process.returncode}），请查看终端日志。')
            self.cleanup()
            self.set_busy(False)
        elif not self.playing:
            self.notice.setText(state.get('text', '正在准备回放…'))

    def cleanup(self):
        self.process = None
        self.cancel_time = None
        if self.temporary:
            self.temporary.cleanup()
            self.temporary = None

    def shutdown(self):
        self.closing = True
        self.timer.stop()
        self.debounce.stop()
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.process:
            self.signal_process(signal.SIGTERM)
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.signal_process(signal.SIGKILL)
                self.process.wait()
            self.signal_process(signal.SIGKILL)
        self.cleanup()
