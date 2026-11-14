"""Per-serial calibration, with no dependency on dense reconstruction settings."""
import math
from pathlib import Path
from ..common.config import read_yaml, resolve_path
from ..common.camera import load_camera
from ..common.paths import CONFIGS
from ..aprilgrid.config import load_config as load_aprilgrid
from ..reconstruction.calibration import calibration
from .roles import ROLES as MULTI_ROLES, reference_for

MULTI_CONFIG = CONFIGS / "multi_vio.yaml"


def roles_for(cfg):
    return tuple(cfg['devices'])


def camera_label(cfg, role):
    sn = cfg['devices'][role]['sn']
    return 'DEMO' if cfg.get('synthetic_only') else sn


def load_multi_config(path=MULTI_CONFIG, *, demo=False, selection=None):
    path = Path(path).expanduser().resolve()
    cfg = read_yaml(path)
    cfg['mode'] = 'multi_vio'
    cfg['synthetic_only'] = demo
    # Accept the old chest setting, but derive the effective reference from
    # enabled roles so an existing config also supports head/hand-only runs.
    if cfg.get('reference_role', 'auto') not in ('auto', 'chest'):
        raise ValueError('reference_role is selected automatically; use auto')
    devices = cfg.get('devices') if selection is None else {r: {'sn': sn} for r,sn in selection.items()}
    if not isinstance(devices, dict) or not devices or not set(devices) <= set(MULTI_ROLES):
        raise ValueError('devices must select 1–4 of chest, head, left_hand and right_hand')
    for role, entry in devices.items():
        if not isinstance(entry, dict) or set(entry) != {'sn'}:
            raise ValueError(f'devices.{role} must contain only sn; calibration follows cameras/<SN>/camera.yaml')
        sn = entry['sn']
        if sn is not None and (not isinstance(sn, str) or not sn.strip() or sn != sn.strip()):
            raise ValueError(f'devices.{role}.sn must be a serial string or null (unassigned)')
    # Live nulls disable a role; demo may intentionally render synthetic roles.
    devices = {r: e for r,e in devices.items() if demo or e['sn'] is not None}
    cfg['reference_role'] = reference_for(devices)
    serials = [v['sn'].lower() for v in devices.values() if v['sn'] is not None]
    if len(serials) != len(set(serials)):
        raise ValueError('Each body part must use a different camera SN')
    # Canonical processing order is independent of YAML order and viewer order.
    cfg['devices'] = {r: devices[r] for r in MULTI_ROLES if r in devices}
    viewer = cfg['viewer']
    order = viewer.get('row_order')
    if (not isinstance(order, list) or len(order) != len(set(order)) or
            not set(order) <= set(MULTI_ROLES) or not set(devices) <= set(order)):
        raise ValueError('viewer.row_order must contain each enabled body part exactly once, without unknown roles')
    viewer['row_order'] = [r for r in order if r in devices]
    viewer['initial_pose_panel'] = len(devices) > 1
    share = viewer.setdefault('initial_pose_share', .14)
    if not isinstance(share, (float, int)) or not math.isfinite(share) or not .08 <= share <= .25:
        raise ValueError('viewer.initial_pose_share must be between 0.08 and 0.25')
    return _load(cfg, path, demo=demo)


def _load(cfg, path, *, demo=False):
    cfg['config_path'] = str(path)
    # Missing key keeps old external configs conservative. The installed
    # Jetson profile explicitly opts into the separate asynchronous budget.
    cfg['timing'].setdefault('board_max_age_s', cfg['timing']['stale_s'])
    for key in ('aprilgrid_config', 'orb_source', 'vocabulary'):
        cfg[key] = str(resolve_path(path.parent, cfg[key]))
        if not Path(cfg[key]).exists():
            raise ValueError(f'Missing {key}: {cfg[key]}')
    for section in ('alignment', 'timing'):
        for key, value in cfg[section].items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f'{section}.{key} must be finite and positive')
    a = cfg['alignment']
    if cfg['timing']['board_max_age_s'] > a['max_duration_s']:
        raise ValueError('timing.board_max_age_s must not exceed alignment.max_duration_s')
    if not 0 < a['min_inlier_fraction'] <= 1 or a['min_duration_s'] >= a['max_duration_s']:
        raise ValueError('Invalid alignment duration or inlier fraction')
    for key in ('min_samples', 'min_tags_per_eye'):
        if type(a[key]) is not int or a[key] < (3 if key == 'min_tags_per_eye' else 5):
            raise ValueError(f'alignment.{key} is too small or not an integer')
    if not 1 <= cfg['fps'] <= 30 or not 0 < cfg['max_imu_gap'] < 1:
        raise ValueError('Invalid fps or IMU gap')
    if type(cfg['features']) is not int or cfg['features'] < 100:
        raise ValueError('features must be an integer >=100')
    v = cfg['viewer']
    v.setdefault('view_distance_m',2.5)
    if sorted(v['display_order']) != ['cam0', 'cam1']:
        raise ValueError('viewer.display_order must contain cam0 and cam1 once')
    if not 1 <= v['jpeg_quality'] <= 100 or v['trajectory_chunk_points'] < 2 or v['max_trajectory_points'] < v['trajectory_chunk_points']:
        raise ValueError('Invalid viewer quality or trajectory limits')
    for key in ('axis_length_m','trajectory_min_distance_m','view_distance_m'):
        if not math.isfinite(v[key]) or v[key]<=0:
            raise ValueError(f'viewer.{key} must be positive')
    if not math.isfinite(cfg['recording']['max_gib']) or cfg['recording']['max_gib'] <= 0:
        raise ValueError('recording.max_gib must be positive')
    devices = {}
    for role in roles_for(cfg):
        sn = cfg['devices'][role]['sn']
        # Explicit demo only: render unassigned cameras using a known optical
        # model. Never substitute another camera's calibration in live mode.
        if sn is None and demo:
            sn = cfg['demo']['calibration_sn']
        try:
            bundle = load_camera(sn=sn)
        except (ValueError, OSError) as exc:
            raise ValueError(f'{role}: {exc}') from exc
        if bundle['reference_camera'] != 'cam1':
            raise ValueError('Native VIO requires cam1 as the reference optical camera')
        native = {**bundle, **{k: cfg[k] for k in ('fps','features','max_imu_gap','orb_source','vocabulary')}}
        calibration(native)  # Validate dimensions and fisheye model before opening USB.
        ag, cameras, board = load_aprilgrid(cfg['aprilgrid_config'], sn=bundle['serial'])
        ag['viewer']['display_isp']['enabled'] = False
        if ag['reference'] != 'cam1':
            raise ValueError('AprilGrid and VIO must use the same cam1 reference')
        devices[role] = dict(native=native, aprilgrid=ag, cameras=cameras, board=board)
    return cfg, devices
