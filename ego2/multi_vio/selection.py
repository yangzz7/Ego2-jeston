"""USB discovery, profile preparation and persistence for the launcher.

Discovery never opens a V4L2 stream. Preferences contain serial numbers only;
calibration always comes from that camera's registered bundle.
"""
from collections import Counter
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from .roles import ROLES, LABELS
from .modes import validate_mode, mode_reference


@dataclass(frozen=True)
class Device:
    sn: str
    node: str
    usb_path: str
    model: str = 'S2 双目'
    problem: str = ''

    @property
    def ready(self):
        return bool(self.sn) and not self.problem


def discover_devices(sysfs='/sys/class/video4linux', dev_root='/dev'):
    """Group by physical USB parent, excluding UVC metadata nodes."""
    physical = {}
    for node in sorted(Path(sysfs).glob('video*')):
        try:
            if (node/'index').read_text().strip() != '0':
                continue
            device = (node/'device').resolve(strict=True)
            for parent in (device, *device.parents):
                if not (parent/'idVendor').is_file():
                    continue
                identity = tuple((parent/key).read_text().strip().lower() for key in ('idVendor','idProduct'))
                if identity == ('04b4', '00f9'):
                    sn = (parent/'serial').read_text().strip() if (parent/'serial').is_file() else ''
                    physical.setdefault(str(parent), (sn, []))[1].append(str(Path(dev_root)/node.name))
                break
        except OSError:
            continue  # USB may disappear between enumeration and reading sysfs.
    counts = Counter(sn.lower() for sn,nodes in physical.values() if sn)
    found = []
    for parent,(sn,nodes) in physical.items():
        problem = ('设备未提供 SN' if not sn else
                   '多台设备使用相同 SN，无法唯一选择' if counts[sn.lower()] > 1 else
                   '同一设备有多个采集节点，无法唯一选择' if len(nodes) != 1 else '')
        found.append(Device(sn, nodes[0], parent, problem=problem))
    return sorted(found, key=lambda d: (d.sn.lower(), d.usb_path))


def scan_devices(config_path, mode='vio'):
    """Prepare missing entries and check optical/IMU parameters without capture."""
    from .profiles import ensure_camera_profile
    from .config import load_multi_config
    from ..common.config import read_yaml
    from ..reconstruction.calibration import write_orb_settings
    # Use a role supported by this layout when probing a single profile.
    # The probe does not assign the physical camera to a body part.
    probe_role = read_yaml(config_path)['viewer']['row_order'][0]
    result = []
    for device in discover_devices():
        if device.problem:
            result.append(device)
            continue
        try:
            ensure_camera_profile(device.sn)
            if mode == 'vio':
                _, profiles = load_multi_config(config_path, selection={probe_role:device.sn})
                with tempfile.TemporaryDirectory(prefix='ego2-profile-check-') as root:
                    write_orb_settings(profiles[probe_role]['native'], Path(root)/'orb.yaml')
            else:
                from .modes import load_mode_config
                load_mode_config(config_path, mode=mode, selection={probe_role:device.sn})
        except Exception as exc:
            device = replace(device, problem=str(exc))
        result.append(device)
    return result


def normalize_selection(selection, *, mode='vio'):
    if not isinstance(selection, dict) or not set(selection) <= set(ROLES):
        raise ValueError('部位分配格式无效')
    selected = {}
    for role in ROLES:
        sn = selection.get(role)
        if sn is None:
            continue
        if not isinstance(sn, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', sn):
            raise ValueError(f'{LABELS[role]}的 SN 格式无效')
        selected[role] = sn
    if len({sn.lower() for sn in selected.values()}) != len(selected):
        raise ValueError('同一 SN 只能分配给一个部位。')
    mode_reference(selected, mode)
    return selected


def validate_connected(selection, devices, *, mode='vio'):
    selected = normalize_selection(selection, mode=mode)
    for role,sn in selected.items():
        matches = [d for d in devices if d.sn == sn]
        if len(matches) != 1:
            raise ValueError(f'{LABELS[role]}：{sn} 未连接或 SN 不唯一，请刷新设备。')
        if not matches[0].ready:
            raise ValueError(f'{LABELS[role]}：{sn} — {matches[0].problem}')
    return selected


def preferences_path(config_path):
    path = Path(config_path).expanduser().resolve()
    return path.with_name(path.stem+'_selection.json')


def read_setup(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('version') not in (1, 2):
            raise ValueError('不支持的文件版本')
        mode = validate_mode(data.get('mode', 'vio'))
        return dict(mode=mode, devices=normalize_selection(data.get('devices'), mode=mode))
    except (ValueError, TypeError) as exc:
        raise ValueError(f'无法读取已保存的相机分配 {path}：{exc}') from exc


def read_preferences(path):
    setup = read_setup(path)
    return setup['devices'] if setup is not None else None


def default_mode(config_path):
    saved = read_setup(preferences_path(config_path))
    if saved is not None:
        return saved['mode']
    from ..common.config import read_yaml
    return validate_mode(read_yaml(Path(config_path)).get('mode', 'vio'))


def write_preferences(path, selection, *, mode='vio'):
    path = Path(path)
    value = dict(version=2, mode=validate_mode(mode), devices=normalize_selection(selection, mode=mode))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.'+path.name, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def default_selection(config_path, *, mode=None):
    """For unattended commands, saved choices take precedence over YAML."""
    saved = read_setup(preferences_path(config_path))
    if saved is not None:
        return normalize_selection(saved['devices'], mode=mode or saved['mode'])
    from ..common.config import read_yaml
    data = read_yaml(Path(config_path).expanduser().resolve())
    return normalize_selection({r:v['sn'] for r,v in data['devices'].items()}, mode=mode or data.get('mode', 'vio'))


def qt_environment():
    env = os.environ.copy()
    # OpenCV wheels export their own Qt5 plugin path; PySide6 uses Qt6.
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR'):
        if '/cv2/' in env.get(key, '').replace('\\','/'):
            env.pop(key)
    return env


def open_launcher(config_path, *, mode=None):
    """Keep Qt's event loop and libraries out of the capture/Rerun process."""
    with tempfile.TemporaryDirectory(prefix='ego2-launcher-') as root:
        result = Path(root)/'selection.json'
        command = [sys.executable, '-m', 'ego2.multi_vio.startup', '--config', str(config_path), '--result', str(result)]
        if mode is not None:
            command += ['--mode', validate_mode(mode)]
        completed = subprocess.run(command, env=qt_environment())
        if completed.returncode == 2:
            return None  # Cancelled: no camera, backend, recording or viewer.
        if completed.returncode:
            raise RuntimeError('Sensia 启动面板异常退出。请检查终端日志、PySide6 和桌面显示环境；无界面运行可用 --no-setup。')
        setup = json.loads(result.read_text(encoding='utf-8'))
        if setup.get('action') == 'replay':
            return dict(action='replay', exit_code=int(setup.get('exit_code', 0)))
        selected_mode = validate_mode(setup['mode'])
        return dict(mode=selected_mode, devices=normalize_selection(setup['devices'], mode=selected_mode))
