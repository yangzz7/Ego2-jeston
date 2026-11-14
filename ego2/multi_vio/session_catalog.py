"""Lightweight saved-session inspection. Never discovers or opens hardware."""
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path

from ..common.paths import OUTPUT
from .modes import MODES
from .storage import FOLDERS

STATES = {'complete': '保存完成', 'preparing': '正在准备录制', 'recording': '正在录制',
          'stopping': '正在停止录制', 'finalizing': '正在保存', 'incomplete': '保存未完成'}
CONTENTS = {'vio': '双目画面、相机轨迹、位姿与标定板标注',
            'gesture': '双目画面、手部骨架与识别标签',
            'aprilgrid': '双目画面、码序号、坐标系与定位结果',
            'reconstruction': '三维点云、双目画面与时间轴'}


@dataclass
class Session:
    path: Path
    mode: str = ''
    format: str = ''
    state: str = ''
    devices: dict = field(default_factory=dict)
    duration: float | None = None
    created: float = 0.
    error: str = ''

    @property
    def playable(self):
        return self.state == 'complete' and not self.error

    @property
    def title(self):
        return MODES.get(self.mode, '未知模式')

    @property
    def state_text(self):
        return STATES.get(self.state, '状态未记录')


def read_object(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'缺少必要文件：{path.name}')
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError(f'元数据文件过大：{path.name}')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'无法读取元数据：{path.name}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'元数据格式无效：{path.name}')
    return value


def inspect_session(directory):
    path = Path(directory).expanduser().resolve()
    item = Session(path)
    try:
        if not path.is_dir():
            raise ValueError('请选择包含 session.json 的数据集文件夹。')
        item.created = path.stat().st_mtime
        session = read_object(path/'session.json')
        item.mode = str(session.get('mode', ''))
        item.state = str(session.get('status', ''))
        try:
            item.created = datetime.fromisoformat(session['created_at']).timestamp()
        except (KeyError, ValueError, TypeError):
            pass
        data = session
        info = None
        if (path/'meta/info.json').exists() or session.get('format') == 'lerobot-v3':
            item.format = 'LeRobot V3'
            info = read_object(path/'meta/info.json')
            data = read_object(path/'meta/ego2.json')
            if info.get('codebase_version') != 'v3.0' or data.get('schema_version') != 2:
                raise ValueError('不支持的数据格式：需要 Ego2 保存的 LeRobot V3 数据。')
            item.mode = str(data.get('mode', item.mode))
        else:
            item.format = 'Rerun'
        if item.mode not in MODES:
            raise ValueError('无法识别数据的功能模式。')
        raw_devices = data.get('devices') or session.get('device_roles') or session.get('devices')
        if not isinstance(raw_devices, dict):
            raw_devices = session.get('config', {}).get('devices', {})
        if isinstance(raw_devices, dict):
            item.devices = {str(role): str(value.get('sn') or value.get('serial') or '未记录')
                            if isinstance(value, dict) else str(value or '未记录')
                            for role, value in raw_devices.items()}
        start, stop = session.get('record_start_monotonic'), session.get('record_stop_monotonic')
        if isinstance(start, (int, float)) and isinstance(stop, (int, float)) and math.isfinite(stop-start) and stop >= start:
            item.duration = stop-start
        elif info is not None:
            fps, frames = info.get('fps'), info.get('total_frames')
            if isinstance(fps, (int, float)) and fps > 0 and isinstance(frames, int) and frames >= 0:
                seconds = frames/fps
                if math.isfinite(seconds): item.duration = seconds
        if item.state != 'complete':
            raise ValueError(item.state_text+'，请等待保存完成或检查录制日志。')
        if info is not None:
            if item.mode not in ('vio', 'gesture'):
                raise ValueError('数据格式与功能模式不匹配。')
            if not item.devices or not isinstance(data.get('config'), dict):
                raise ValueError('数据缺少录制时的相机分配或布局。')
            for glob in ('data/chunk-*/*.parquet', 'meta/episodes/chunk-*/*.parquet', 'videos/*/chunk-*/*.mp4'):
                if next(path.glob(glob), None) is None:
                    raise ValueError('缺少必要的数据、视频或分段索引文件。')
        elif item.mode not in ('aprilgrid', 'reconstruction'):
            raise ValueError('该模式需要完整的 LeRobot V3 数据集。')
        elif not (path/'replay.rrd').is_file() or (path/'replay.rrd').stat().st_size == 0:
            raise ValueError('缺少可读取的 replay.rrd 文件。')
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        item.error = str(exc)
    return item


def list_sessions(root=OUTPUT):
    items = []
    for folder in sorted(set(FOLDERS.values())):
        directory = Path(root)/folder
        if not directory.exists(): continue
        for path in directory.iterdir():
            if path.is_dir(): items.append(inspect_session(path))
    return sorted(items, key=lambda item: (item.created, item.path.name), reverse=True)


def duration_text(seconds):
    if seconds is None: return '未记录'
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'
