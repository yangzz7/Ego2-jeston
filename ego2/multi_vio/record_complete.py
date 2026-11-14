"""Success notification in a separate Qt process, leaving live preview running."""
import argparse
from pathlib import Path

# Reuse the launcher's OpenCV/Qt library and plugin setup before creating widgets.
from .launcher import (QApplication, QDialog, QHBoxLayout, QPushButton,
    QVBoxLayout, QWidget, Qt, QTimer, QPainter, QPen, QColor, QPointF, STYLE, label)
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QTextOption
from PySide6.QtWidgets import QTextEdit


class SuccessIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(52, 52)
        self.setAccessibleName('保存成功')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#43bf8c'))
        painter.drawEllipse(2, 2, 48, 48)
        painter.setPen(QPen(QColor('#ffffff'), 3.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPolyline([QPointF(15, 26), QPointF(23, 34), QPointF(37, 19)])


class RecordingCompleteDialog(QDialog):
    def __init__(self, path, channel=None):
        super().__init__()
        self.path = Path(path)
        self.channel = Path(channel) if channel is not None else None
        self.setWindowTitle('Data Collector')
        self.setStyleSheet(STYLE + '''
QTextEdit#savedPath { background:#282c32; border:1px solid #424953;
    border-radius:7px; padding:12px; font-size:15px; }
''')
        self.resize(720, 340)
        self.setMinimumWidth(570)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 26, 30, 24)
        layout.setSpacing(14)
        heading = QHBoxLayout()
        heading.setSpacing(18)
        heading.addWidget(SuccessIcon())
        text = QVBoxLayout()
        text.setSpacing(4)
        text.addWidget(label('录制结束', 'title'))
        text.addWidget(label('数据已保存，Rerun继续预览数据。', 'sub'))
        heading.addLayout(text, 1)
        layout.addLayout(heading)
        layout.addSpacing(4)
        layout.addWidget(label('保存位置', 'section'))
        self.location = QTextEdit()
        self.location.setObjectName('savedPath')
        self.location.setAccessibleName('保存位置')
        self.location.setReadOnly(True)
        self.location.setPlainText(str(self.path))
        self.location.setWordWrapMode(QTextOption.WrapAnywhere)
        self.location.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.location.setMinimumHeight(84)
        layout.addWidget(self.location, 1)
        self.notice = label('', 'warning')
        self.notice.setWordWrap(True)
        self.notice.hide()
        layout.addWidget(self.notice)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.open_folder = QPushButton('打开文件夹')
        self.open_folder.setAutoDefault(False)
        self.open_folder.clicked.connect(self.show_folder)
        self.confirm = QPushButton('确定')
        self.confirm.setObjectName('start')
        self.confirm.setDefault(True)
        self.confirm.clicked.connect(self.accept)
        buttons.addWidget(self.open_folder)
        buttons.addWidget(self.confirm)
        layout.addLayout(buttons)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        if self.channel is not None:
            self.timer.start(100)

    def show_folder(self):
        if not self.path.is_dir():
            self.notice.setText('保存文件夹当前不可访问，请检查磁盘是否已连接。')
            self.notice.show()
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path))):
            self.notice.setText('无法打开文件管理器，可复制上方路径手动查看。')
            self.notice.show()
        else:
            self.notice.hide()

    def poll(self):
        request = self.channel/'raise-completion'
        if request.exists():
            request.unlink(missing_ok=True)
            self.showNormal()
            self.raise_()
            self.activateWindow()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--channel')
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    dialog = RecordingCompleteDialog(args.path, args.channel)
    QTimer.singleShot(0, lambda: (dialog.raise_(), dialog.activateWindow(), dialog.confirm.setFocus()))
    dialog.exec()
    return 0  # Confirm, Escape and the window close button all dismiss normally.


if __name__ == '__main__':
    raise SystemExit(main())
