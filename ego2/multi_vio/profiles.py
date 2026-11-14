"""Create missing S2 camera entry files from a unit's supplied calibration."""
import math
import os
from pathlib import Path
import re
import tempfile

import yaml

from ..common import camera as camera_profiles
from ..common.config import load_calibration, read_yaml, se3
from ..reconstruction.calibration import write_orb_settings


def _validate_parameters(directory, imu_name):
    for filename in ('camchain.yaml', 'camchain-imucam.yaml'):
        path = directory/filename
        cameras = load_calibration(path)
        if set(cameras) != {'cam0', 'cam1'} or any(cam.size != (1280,1024) for cam in cameras.values()):
            raise ValueError(f'{filename} 必须包含 1280×1024 的 cam0 和 cam1 双目鱼眼参数')
    raw = read_yaml(directory/'camchain-imucam.yaml')
    for name in ('cam0', 'cam1'):
        se3(raw[name]['T_cam_imu'])
        if not math.isfinite(float(raw[name]['timeshift_cam_imu'])):
            raise ValueError(f'{name}.timeshift_cam_imu 必须为有限数值')
    imu = read_yaml(directory/imu_name)['imu0']
    for key in ('accelerometer_noise_density', 'accelerometer_random_walk',
                'gyroscope_noise_density', 'gyroscope_random_walk', 'update_rate'):
        value = imu[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f'{imu_name}: imu0.{key} 必须为正的有限数值')
    if not float(imu['update_rate']).is_integer():
        raise ValueError(f'{imu_name}: imu0.update_rate 必须为整数频率')
    # Exercise the actual ORB serializer before publishing an entry file.
    with tempfile.TemporaryDirectory(prefix='ego2-auto-profile-check-') as root:
        write_orb_settings(dict(calibration=str(directory/'camchain-imucam.yaml'),
            imu_calibration=str(directory/imu_name), fps=30, features=1600), Path(root)/'orb.yaml')


def ensure_camera_profile(sn):
    """Return (path, created); an existing entry is never rewritten.

    The transport defaults are the same S2 defaults as the registered devices.
    No calibration values are copied from a different SN or synthesized.
    """
    if not isinstance(sn, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', sn):
        raise ValueError('设备 SN 格式无效')
    directory = camera_profiles.CAMERAS/sn
    target = directory/'camera.yaml'
    if target.is_file():
        return target, False
    if target.exists() or target.is_symlink():
        raise ValueError(f'{target} 已存在但不是可读取的配置文件，请检查该路径')
    # Accept the legacy filename too, but prefer the user's imu.yaml.
    imu_name = 'imu.yaml' if (directory/'imu.yaml').is_file() else 'imu-calibrated.yaml'
    missing = [name for name in ('camchain.yaml','camchain-imucam.yaml') if not (directory/name).is_file()]
    if not (directory/imu_name).is_file():
        missing.append('imu.yaml（或 imu-calibrated.yaml）')
    if missing:
        raise ValueError(f'{sn} 无法自动生成 camera.yaml：缺少 {", ".join(missing)}；请放入 {directory}')
    try:
        _validate_parameters(directory, imu_name)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, yaml.YAMLError) as exc:
        raise ValueError(f'{sn} 无法自动生成 camera.yaml：标定参数无效（{exc}）') from exc
    profile = dict(serial=sn, calibration='camchain-imucam.yaml', imu_calibration=imu_name,
        reference_camera='cam1', acceleration_scale=9.7887,
        input=dict(type='cyper', device='auto', width=1280, height=1024, fps=30,
                   bayer='auto', white_balance=True),
        channels=[dict(camera='cam0', label='C1 / cam0', invalid_bottom_rows=2),
                  dict(camera='cam1', label='C2 / cam1', invalid_bottom_rows=2)])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                         prefix='.camera-', suffix='.yaml', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write('# Auto-generated Cyperstereo S2 entry; calibration belongs to this SN.\n')
            yaml.safe_dump(profile, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish a complete file without overwriting one created concurrently.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not target.is_file():
                raise ValueError(f'{target} 已被其他文件或目录占用')
            return target, False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'[camera] 已自动生成 {target}', flush=True)
    return target, True


def prepare_camera_profiles():
    """Prepare all SN directories, including disconnected units, at startup.

    Return per-unit problems instead of blocking other usable cameras. The
    connected-device scanner also retries so files added in the panel work.
    """
    created, problems = [], {}
    for directory in sorted(camera_profiles.CAMERAS.iterdir()):
        if not directory.is_dir() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', directory.name):
            continue
        try:
            path, generated = ensure_camera_profile(directory.name)
            if generated:
                created.append(path)
        except (OSError, ValueError) as exc:
            problems[directory.name] = str(exc)
    return created, problems
