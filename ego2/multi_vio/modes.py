"""Mode policy and per-SN configuration for the unified launcher."""
from pathlib import Path
import copy
import math

from ..common.config import read_yaml, resolve_path
from ..common.paths import CONFIGS
from .roles import reference_for

MODES = {
    'rgb': '纯RGB画面',
    'vio': '单/多相机轨迹',
    'gesture': '手势识别',
    'aprilgrid': 'AprilGrid 标定板跟踪与定位',
    'reconstruction': '三维重建',
}
HINTS = {
    'rgb': '仅显示所选相机的双目增强 RGB 画面和时间轴；支持单台或多台相机。可拖动时间轴查看本次预览，Ctrl+C 退出。',
    'vio': '启动 Rerun 后，请缓慢平移并适当旋转相机，确保各相机视野内始终有丰富、清晰的纹理，以完成 VIO 初始化。\n初始化完成后，仍需在纹理丰富的环境中运行，再观测同一块固定标定板完成初始对齐。',
    'gesture': '请让手部清晰地出现在相机视野中，保持充足光照。各相机独立识别手势，无需 VIO 初始化或标定板对齐。',
    'aprilgrid': '请让标定板清晰地出现在相机视野中，以完成识别与定位。各相机独立计算相对标定板的位姿，无需 VIO 初始化。',
    'reconstruction': '请选择一台相机，缓慢平移并适当旋转，在纹理丰富的环境中完成 VIO 初始化。Rerun 沿用原三维重建布局：左侧 3D 点云，右侧双目 RGB 画面。',
}


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown mode {mode!r}; choose {", ".join(MODES)}')
    return mode


def mode_reference(roles, mode):
    validate_mode(mode)
    if mode == 'vio':
        return reference_for(roles)
    if not roles:
        raise ValueError('请至少选择一台相机。')
    if mode == 'reconstruction' and len(roles) != 1:
        raise ValueError('三维重建仅支持一台相机，请只保留一个部位，其余选择“不启用”。')
    return None


def load_mode_config(path, *, mode='vio', selection=None, demo=False):
    validate_mode(mode)
    if mode == 'vio':
        from .config import load_multi_config
        return load_multi_config(path, demo=demo, selection=selection)
    from .selection import normalize_selection
    from .roles import ROLES
    path = Path(path).expanduser().resolve()
    cfg = read_yaml(path)
    choices = selection if selection is not None else {r: e['sn'] for r, e in cfg['devices'].items()}
    # Demo-only nulls explicitly stand for synthetic devices, as in VIO demo.
    if demo:
        roles = tuple(choices)
        if not roles or not set(roles) <= set(ROLES):
            raise ValueError('Invalid demo body roles')
        mode_reference(roles, mode)
        assigned = {r: sn for r, sn in choices.items() if sn is not None}
        if assigned:
            normalize_selection(assigned, mode=mode)
    else:
        choices = normalize_selection(choices, mode=mode)
    cfg.update(mode=mode, synthetic_only=demo, config_path=str(path), reference_role=None,
               devices={r: {'sn': choices[r]} for r in ROLES if r in choices})
    order = cfg['viewer']['row_order']
    if (not isinstance(order, list) or len(order) != len(set(order)) or
            not set(order) <= set(ROLES) or not set(choices) <= set(order)):
        raise ValueError('viewer.row_order must contain each enabled body part exactly once')
    cfg['viewer']['row_order'] = [r for r in order if r in choices]
    cfg['viewer']['initial_pose_panel'] = False
    limit = cfg['recording']['max_gib']
    if isinstance(limit, bool) or not isinstance(limit, (float, int)) or not math.isfinite(limit) or limit <= 0:
        raise ValueError('recording.max_gib must be finite and positive')
    profile_path = resolve_path(path.parent, cfg.get(mode + '_config', str(CONFIGS / (mode + '.yaml'))))
    cfg[mode + '_config'] = str(profile_path)
    if mode == 'rgb':
        settings = read_yaml(profile_path)
        for key in ('fps', 'max_fps'):
            value = settings.get(key, 30)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'rgb.{key} must be finite and positive')
            settings[key] = float(value)
        quality = settings.get('jpeg_quality', 85)
        if isinstance(quality, bool) or not isinstance(quality, int) or not 0 <= quality <= 100:
            raise ValueError('rgb.jpeg_quality must be 0..100')
        cfg['viewer']['jpeg_quality'] = quality
        cfg['viewer']['display_isp'] = {**cfg['viewer'].get('display_isp', {}),
                                      'enabled': True, 'max_fps': settings['max_fps']}
        cfg['fps'] = settings['fps']
    devices = {}
    for role in cfg['devices']:
        sn = choices[role] or cfg['demo']['calibration_sn']
        if mode == 'rgb':
            from ..common.config import load_profile
            profile, cameras = load_profile(profile_path, sn=sn)
            if profile['input']['type'] != 'cyper':
                raise ValueError('纯RGB画面目前支持 Cyperstereo 双目相机')
            profile['input'].update(fps=cfg['fps'], preserve_display_raw=True,
                                    preserve_raw=False, white_balance=False)
            devices[role] = dict(profile=profile, cameras=cameras, native={'serial': sn})
        elif mode == 'gesture':
            from ..gesture.config import load_config
            profile, cameras = load_config(profile_path, sn=sn)
            if not Path(profile['gesture']['model']).is_file():
                raise ValueError(f'Missing gesture model: {profile["gesture"]["model"]}')
            devices[role] = dict(profile=profile, cameras=cameras, native={'serial': sn})
        elif mode == 'aprilgrid':
            from ..aprilgrid.config import load_config
            profile, cameras, board = load_config(profile_path, sn=sn)
            devices[role] = dict(profile=profile, cameras=cameras, board=board, native={'serial': sn})
        else:
            from ..reconstruction.calibration import load_config, write_orb_settings
            import tempfile
            profile = load_config(profile_path, sn=sn)
            profile['viewer']['display_isp'] = copy.deepcopy(cfg['viewer'].get('display_isp', {}))
            # Display config paths follow the Sensia config, not calibration.
            profile['config_path'] = str(path)
            with tempfile.TemporaryDirectory(prefix='ego2-mode-check-') as root:
                write_orb_settings(profile, Path(root) / 'orb.yaml')
            devices[role] = dict(profile=profile, native=profile)
    return cfg, devices
