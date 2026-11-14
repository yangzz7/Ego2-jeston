"""Optional chest in single-camera sessions, required chest for multiple cameras."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import rerun.blueprint as blueprint_types
import yaml

from ego2.multi_vio.alignment import Coordinator, inverse
from ego2.multi_vio.config import MULTI_CONFIG, MULTI_ROLES, load_multi_config
from ego2.multi_vio.recording import Recording
from ego2.multi_vio.selection import (
    Device, default_selection, normalize_selection, preferences_path,
    scan_devices, write_preferences,
)
from ego2.multi_vio.visualizer import build_blueprint, initial_multi_eye_geometry
from tests.test_multi_vio_core import measured, transform


def raw_config():
    cfg = yaml.safe_load(MULTI_CONFIG.read_text())
    for key in ('aprilgrid_config', 'orb_source', 'vocabulary'):
        cfg[key] = str((MULTI_CONFIG.parent / cfg[key]).resolve())
    return cfg


class ReferencePolicyTests(unittest.TestCase):
    def test_every_role_combination_has_same_rule_in_selection_and_yaml(self):
        serials = dict(zip(MULTI_ROLES, ('s200100', 's200032', 's200097', 's200098')))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'config.yaml'
            for count in range(5):
                for roles in itertools.combinations(MULTI_ROLES, count):
                    with self.subTest(roles=roles):
                        choice = {r: serials[r] for r in roles}
                        raw = raw_config()
                        raw['devices'] = {r: dict(sn=choice.get(r)) for r in MULTI_ROLES}
                        path.write_text(yaml.safe_dump(raw))
                        if not roles or (count > 1 and 'chest' not in roles):
                            with self.assertRaises(ValueError):
                                normalize_selection(choice)
                            with patch('ego2.multi_vio.config.load_camera') as camera:
                                with self.assertRaises(ValueError):
                                    load_multi_config(path)
                                camera.assert_not_called()
                            continue
                        self.assertEqual(normalize_selection(choice), choice)
                        cfg, devices = load_multi_config(path)
                        self.assertEqual(cfg['reference_role'], roles[0] if count == 1 else 'chest')
                        self.assertEqual(set(devices), set(roles))
                        self.assertEqual(set(cfg['viewer']['row_order']), set(roles))
                        self.assertEqual(cfg['viewer']['initial_pose_panel'], count > 1)

    def test_old_config_and_saved_non_chest_choice_work_without_panel(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'config.yaml'
            raw = raw_config()
            raw['reference_role'] = 'chest'  # Previously required in YAML.
            path.write_text(yaml.safe_dump(raw))
            for role in ('head', 'left_hand', 'right_hand'):
                with self.subTest(role=role):
                    choice = {role: 's200032'}
                    write_preferences(preferences_path(path), choice)
                    selected = default_selection(path)
                    cfg, devices = load_multi_config(path, selection=selected)
                    self.assertEqual(selected, choice)
                    self.assertEqual(tuple(devices), (role,))
                    self.assertEqual(cfg['reference_role'], role)

    def test_scanning_accepts_single_role_layout_without_chest(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'config.yaml'
            raw = raw_config()
            raw['devices'] = {'head': dict(sn='s200032')}
            raw['viewer']['row_order'] = ['head']
            path.write_text(yaml.safe_dump(raw))
            self.assertEqual(default_selection(path), {'head': 's200032'})
            with patch('ego2.multi_vio.selection.discover_devices',
                       return_value=[Device('s200032', '/dev/video0', '/usb/0')]):
                devices = scan_devices(path)
            self.assertTrue(devices[0].ready, devices[0].problem)

    def test_live_start_and_build_use_selected_reference_profile(self):
        from ego2.multi_vio.app import run
        from ego2.multi_vio.__main__ import main
        for role in ('head', 'left_hand', 'right_hand'):
            with self.subTest(role=role), contextlib.redirect_stdout(io.StringIO()):
                choice = {role: 's200032'}
                cfg, devices = load_multi_config(selection=choice)
                args = SimpleNamespace(demo=False, jobs=2, rebuild=False, record=False, no_viz=True)
                with patch('ego2.multi_vio.transport.check_devices') as check, \
                     patch('ego2.multi_vio.app.ensure_native', return_value='/fake/backend') as native, \
                     patch('ego2.multi_vio.app.live') as live:
                    run(cfg, devices, args)
                    check.assert_called_once_with(devices)
                    native.assert_called_once_with(devices[role]['native'], 2, False)
                    live.assert_called_once()
                with patch('ego2.multi_vio.selection.default_selection', return_value=choice), \
                     patch('ego2.multi_vio.profiles.prepare_camera_profiles', return_value=({}, {})), \
                     patch('ego2.multi_vio.transport.ensure_native', return_value='/fake/backend') as native:
                    self.assertEqual(main(['build', '--no-setup', '--mode', 'vio']), 0)
                    self.assertEqual(native.call_args.args[0]['serial'], 's200032')


class SingleCameraGeometryTests(unittest.TestCase):
    def test_each_role_aligns_continues_without_board_and_records_own_reference(self):
        for role in MULTI_ROLES:
            with self.subTest(role=role):
                cfg, devices = load_multi_config(selection={role: 's200032'})
                cfg['alignment'].update(min_samples=5, min_duration_s=.3)
                coordinator = Coordinator(cfg, devices)
                board_camera = transform((.15, -.3, .2), (.2, -.1, .7))
                anchor = transform((.4, -.1, -.35), (1., .2, -.5))
                local = inverse(anchor) @ board_camera

                def packet(index, pose):
                    return dict(frame_id=index, timestamp=index*.1, capture_time=index*.1,
                                map_id='0', initialization_complete=True, valid=True,
                                T_world_camera=pose.tolist())

                # A valid VIO pose alone must not bypass board initialization.
                p = packet(0, local)
                coordinator.ingest(role, p, 0.)
                coordinator.relative(0.)
                self.assertIsNone(coordinator.pose(role, 0.))
                for index in range(6):
                    p = packet(index, local)
                    coordinator.ingest(role, p, index*.1)
                    coordinator.observe_board(role, p, measured(board_camera), index*.1)
                    coordinator.relative(index*.1)
                self.assertTrue(coordinator.started)
                self.assertEqual(coordinator.reference, role)
                self.assertEqual(coordinator.initial_relatives, {})
                self.assertEqual(coordinator.initial_deltas, {})
                np.testing.assert_allclose(coordinator.shared_from_board[:3, :3], anchor[:3, :3].T, atol=1e-9)
                np.testing.assert_allclose(coordinator.shared_from_board[:3, 3], 0., atol=1e-9)
                initial = coordinator.pose(role, .5)
                np.testing.assert_allclose(initial, coordinator.shared_from_board @ board_camera, atol=1e-9)

                # Moving up in the VIO frame remains world +Z after board loss.
                local[:3, 3] += [0., 0., .25]
                p = packet(6, local)
                coordinator.ingest(role, p, .6)
                coordinator.observe_board(role, p, None, .6)
                coordinator.relative(.6)
                pose = coordinator.pose(role, .6)
                np.testing.assert_allclose(pose[:3, 3] - initial[:3, 3], [0., 0., .25], atol=1e-9)
                event = next(e for e in coordinator.events if e['event'] == 'started')
                self.assertEqual(event['reference_role'], role)

                blueprint = build_blueprint(cfg)
                left, right = blueprint.root_container.contents
                self.assertEqual(len(right.contents), 1)
                self.assertEqual(len(right.contents[0].contents), 2)
                self.assertFalse(cfg['viewer']['initial_pose_panel'])
                self.assertIsInstance(left, blueprint_types.Spatial3DView)
                eye = initial_multi_eye_geometry(coordinator.initial_poses, cfg['viewer'])
                forward = eye['look_target'] - eye['position']
                np.testing.assert_allclose(forward / np.linalg.norm(forward), initial[:3, 2], atol=1e-9)

                with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
                    args = SimpleNamespace(record=True, session=str(Path(root) / 'session'), demo=False)
                    recording = Recording(cfg, devices, args)
                    recording.event(event)
                    p={**p,'host_monotonic':.6}
                    recording.pose(role, p, pose, coordinator.board_pose(role, .6))
                    recording.demo_capture(role,p,(np.zeros((16,24,3),np.uint8),)*2)
                    recording.finish(coordinator, {role: {}})
                    session = Path(args.session)
                    metadata = json.loads((session / 'session.json').read_text())
                    self.assertTrue(metadata['finished'])
                    self.assertEqual(metadata['reference_role'], role)
                    self.assertEqual(metadata['reference_serial'], 's200032')
                    self.assertEqual(metadata['initial_T_reference_camera'], {})
                    self.assertEqual(metadata['initial_deltas'], {})
                    self.assertEqual(set(metadata['initial_T_shared_camera']), {role})
                    self.assertEqual({p.name for p in (session/'calibration').iterdir() if p.is_dir()}, {role})
                    self.assertFalse((session/'_work').exists())
                    from ego2.multi_vio.replay import Reader
                    saved_pose=list(Reader(session).frames())[0][2]
                    np.testing.assert_allclose(np.asarray(saved_pose[f'observation.{role}.shared_pose']).reshape(4,4), pose)


if __name__ == '__main__':
    unittest.main()
