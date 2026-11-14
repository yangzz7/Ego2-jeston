"""Unified mode selection, complete-frame transport and 2D-only layouts."""
from contextlib import redirect_stdout
import io
import itertools
import json
from pathlib import Path
import pickle
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as b
import yaml

from ego2.common.sources import Pair
from ego2.multi_vio.config import MULTI_CONFIG
from ego2.multi_vio.roles import ROLES
from ego2.multi_vio.modes import MODES, load_mode_config
from ego2.multi_vio.selection import (normalize_selection, read_setup, write_preferences,
                                     default_selection, default_mode, preferences_path)
from ego2.multi_vio.mode_viewer import (BatchRecording, GestureSink,
                                      UnifiedViewer, build_blueprint)
from ego2.multi_vio.multi_mode import WorkerStop, worker_arguments


class ModeConfigTests(unittest.TestCase):
    def test_all_subsets_follow_mode_specific_rule(self):
        for mode in MODES:
            for count in range(5):
                for roles in itertools.combinations(ROLES, count):
                    choice = {r: f's20000{i}' for i,r in enumerate(roles)}
                    with self.subTest(mode=mode, roles=roles):
                        invalid = (not roles or (mode == 'vio' and count > 1 and 'chest' not in roles)
                                   or (mode == 'reconstruction' and count != 1))
                        if invalid:
                            with self.assertRaises(ValueError): normalize_selection(choice, mode=mode)
                        else:
                            self.assertEqual(normalize_selection(choice, mode=mode), choice)
            with self.assertRaisesRegex(ValueError, '同一 SN'):
                normalize_selection({'head':'s200032','left_hand':'S200032'}, mode=mode)

    def test_modes_load_own_profiles_without_chest_and_only_two_dimensional_views(self):
        for mode in ('gesture', 'aprilgrid'):
            with self.subTest(mode=mode):
                cfg, devices = load_mode_config(MULTI_CONFIG, mode=mode,
                    selection={'head':'s200032','left_hand':'s200100'})
                self.assertIsNone(cfg['reference_role'])
                self.assertEqual([d['native']['serial'] for d in devices.values()], ['s200032','s200100'])
                blueprint = build_blueprint(cfg, imu=False)  # Original image-only/replay layout.
                self.assertEqual(len(blueprint.root_container.contents), 2)
                for row in blueprint.root_container.contents:
                    self.assertEqual(len(row.contents), 2)
                    self.assertTrue(all(isinstance(v, b.Spatial2DView) for v in row.contents))
                    self.assertTrue(row.contents[0].origin.endswith('/cam1'))
                    self.assertTrue(row.contents[1].origin.endswith('/cam0'))
                with tempfile.TemporaryDirectory() as root:
                    blueprint.save('test-'+mode, str(Path(root)/'layout.rbl'))

    def test_optical_modes_do_not_load_vio_or_require_orb_files(self):
        raw = yaml.safe_load(MULTI_CONFIG.read_text())
        for key in ('gesture_config','aprilgrid_config'):
            raw[key] = str((MULTI_CONFIG.parent/raw[key]).resolve())
        raw.update(orb_source='/not-installed', vocabulary='/not-installed')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'config.yaml'
            path.write_text(yaml.safe_dump(raw))
            for mode in ('gesture','aprilgrid'):
                with patch('ego2.reconstruction.calibration.write_orb_settings', side_effect=AssertionError):
                    load_mode_config(path, mode=mode, selection={'head':'s200032'})

    def test_old_preferences_migrate_and_explicit_mode_override_revalidates(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = Path(root)/'config.yaml'
            cfg.write_text('devices: {head: {sn: s200032}}\nmode: gesture\n')
            path = preferences_path(cfg)
            path.write_text(json.dumps(dict(version=1,devices={'chest':'s200100'})))
            self.assertEqual(read_setup(path)['mode'], 'vio')
            choice = {'head':'s200032','left_hand':'s200100'}
            write_preferences(path, choice, mode='gesture')
            self.assertEqual(default_mode(cfg), 'gesture')
            self.assertEqual(default_selection(cfg), choice)
            with self.assertRaisesRegex(ValueError, '胸部'):
                default_selection(cfg, mode='vio')
            with self.assertRaisesRegex(ValueError, '仅支持一台'):
                default_selection(cfg, mode='reconstruction')
            self.assertEqual(json.loads(path.read_text())['version'], 2)

    def test_cli_dispatches_panel_mode_and_noninteractive_mode(self):
        from ego2.multi_vio.__main__ import main
        choice = {'head':'s200032','left_hand':'s200100'}
        for mode in ('gesture','aprilgrid','reconstruction'):
            selected = {'head':'s200032'} if mode == 'reconstruction' else choice
            for gui in (True, False):
                with self.subTest(mode=mode, gui=gui), redirect_stdout(io.StringIO()), \
                     patch('ego2.multi_vio.profiles.prepare_camera_profiles',return_value=([],{})), \
                     patch('ego2.multi_vio.selection.open_launcher',return_value=dict(mode=mode,devices=selected)) as panel, \
                     patch('ego2.multi_vio.selection.default_selection',return_value=selected), \
                     patch('ego2.multi_vio.app.run') as vio, \
                     patch('ego2.multi_vio.multi_mode.run') as run:
                    argv = ['--no-record'] if gui else ['--mode', mode, '--no-setup', '--no-record']
                    self.assertEqual(main(argv), 0)
                    self.assertEqual(run.call_args.args[0]['mode'], mode)
                    self.assertEqual(set(run.call_args.args[1]), set(selected))
                    self.assertFalse(run.call_args.args[2].record)
                    vio.assert_not_called()
                    self.assertEqual(panel.called, gui)


class PreviewTests(unittest.TestCase):
    def test_preview_transport_preserves_arrow_types_required_by_rerun(self):
        # A changed nested field nullability makes Rerun skip otherwise valid
        # points/lines, including the text labels anchored by Points2D.
        q = queue.Queue()
        rec = BatchRecording(q, 'head', 0., demo=True, eye_names=('cam0', 'cam1'))
        archetypes = [
            rr.Points2D([[10., 20.], [30., 40.]], labels=['Left', 'Right'],
                        colors=[75, 210, 255], radii=3, draw_order=20),
            rr.LineStrips2D([[[10., 20.], [30., 40.]], [[5., 6.], [7., 8.]]],
                           colors=[[255, 0, 0], [0, 255, 0]], radii=1.5),
            rr.Clear(recursive=True),
            rr.TextDocument('camera metadata'),
            rr.Image(np.full((8, 8, 3), 42, np.uint8)),
        ]
        rec.begin(1., sequence=30)
        for index, archetype in enumerate(archetypes):
            rec.log(f'/head/eye0/item{index}', archetype)
        rec.finish()
        logs = pickle.loads(q.get_nowait())['logs']
        for original, (_, items, _, _) in zip(archetypes, logs):
            before = original.as_component_batches()
            after = items[0].as_component_batches()
            self.assertEqual(len(before), len(after))
            for a, b in zip(before, after):
                with self.subTest(component=str(a.component_descriptor())):
                    self.assertEqual(a.component_descriptor(), b.component_descriptor())
                    self.assertEqual(a.as_arrow_array().type, b.as_arrow_array().type)
                    self.assertTrue(a.as_arrow_array().equals(b.as_arrow_array()))

    def test_serialized_stereo_batches_preserve_paths_and_common_receipt_time(self):
        q = queue.Queue(maxsize=1)
        rec = BatchRecording(q, 'left_hand', 100., eye_names=('cam0','cam1'))
        rec.begin(2., received=107., sequence=9)
        rec.set_time('capture_time', duration=2.)
        rec.set_time('frame', sequence=9)
        rec.log('/head/eye0/rgb', rr.TextDocument('left transport lane'))
        rec.log('/head/eye1/rgb', rr.TextDocument('right transport lane'))
        rec.finish()
        # Drop an entire new stereo frame when the preview queue is full.
        rec.begin(3., received=108., sequence=10)
        rec.log('/head/eye0/rgb', rr.TextDocument('new'))
        rec.log('/head/eye1/rgb', rr.TextDocument('new'))
        rec.finish()
        self.assertEqual((rec.sent,rec.dropped),(1,1))
        batch = pickle.loads(q.get_nowait())
        self.assertEqual([l[0] for l in batch['logs']], ['/cameras/left_hand/cam0/rgb','/cameras/left_hand/cam1/rgb'])
        self.assertTrue(all(l[3]['capture_time']=={'duration':7.} for l in batch['logs']))
        viewer = UnifiedViewer.__new__(UnifiedViewer)
        viewer.recording = MagicMock()
        viewer.consume(pickle.dumps(batch))
        self.assertEqual(viewer.recording.reset_time.call_count,2)
        self.assertEqual(viewer.recording.log.call_count,2)

    def test_gesture_annotations_follow_correct_eye_and_clear_disappeared_hands(self):
        q = queue.Queue(maxsize=4)
        rec = BatchRecording(q,'head',0.,demo=True,eye_names=('cam0','cam1'))
        sink = GestureSink({'viewer':{'jpeg_quality':85}},rec)
        hand = SimpleNamespace(id=7,handedness='Right',gesture='Open_Palm',score=.95,pinch='released',
                               points=np.full((21,2),.2)*100,visible=np.ones(21,dtype=bool))
        pair = Pair((np.full((100,100,3),40,np.uint8), np.full((100,100,3),180,np.uint8)),0.,0.,0)
        sink.log(pair,0.,([hand],[]))
        logs = pickle.loads(q.get())['logs']
        paths = [l[0] for l in logs]
        self.assertIn('/cameras/head/cam0/hands/7/points',paths)
        self.assertNotIn('/cameras/head/cam1/hands/7/points',paths)
        points = next(items[0] for p,items,_,_ in logs if p.endswith('/7/points'))
        self.assertEqual(len(points.positions.as_arrow_array()),21)
        sink.log(pair,.1,([],[]))
        logs = pickle.loads(q.get())['logs']
        clears = [p for p,items,_,_ in logs if any(isinstance(i,rr.Clear) for i in items)]
        self.assertIn('/cameras/head/cam0/hands',clears)
        self.assertIn('/cameras/head/cam1/hands',clears)

    def test_reconstruction_is_single_camera_and_reuses_original_3d_viewer(self):
        from ego2.multi_vio.multi_mode import run
        with self.assertRaisesRegex(ValueError,'仅支持一台'):
            load_mode_config(MULTI_CONFIG,mode='reconstruction',
                             selection={'head':'s200032','left_hand':'s200100'})
        cfg, devices = load_mode_config(MULTI_CONFIG,mode='reconstruction',selection={'left_hand':'s200032'})
        blueprint = build_blueprint(cfg)
        cloud, images = blueprint.root_container.contents
        self.assertIsInstance(cloud,b.Spatial3DView)
        self.assertEqual(len(images.contents),2)
        self.assertTrue(all(isinstance(v,b.Spatial2DView) for v in images.contents))
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            args=SimpleNamespace(mode='reconstruction',record=True,session=str(Path(root)/'session'),
                                 demo=True,frames=3,no_viz=True,headless=True,rebuild=False,jobs=2)
            with patch('ego2.reconstruction.app.run') as original, patch('ego2.multi_vio.multi_mode.finish_replay'):
                run(cfg,devices,args)
                original.assert_called_once()
                profile, options = original.call_args.args
                self.assertEqual(profile['serial'],'s200032')
                self.assertEqual(original.call_args.kwargs,{})  # No alternate viewer.
                self.assertTrue(options.session.endswith('/left_hand/data'))

    def test_normal_worker_completion_does_not_stop_sibling(self):
        master = threading.Event()
        first, second = WorkerStop(master), WorkerStop(master)
        first.set()
        self.assertTrue(first.is_set())
        self.assertFalse(master.is_set())
        self.assertFalse(second.is_set())
        master.set()
        self.assertTrue(second.wait(.1))

    def test_worker_recording_and_preview_flags_are_independent(self):
        args = SimpleNamespace(record=False,no_viz=False,frames=5)
        opts = worker_arguments(args,Path('/tmp/example'))
        self.assertIsNone(opts.dataset)
        self.assertIsNone(opts.save)
        self.assertFalse(opts.no_viz)
        args.record=True
        opts = worker_arguments(args,Path('/tmp/example'))
        self.assertEqual(opts.dataset,Path('/tmp/example/data'))
        self.assertIsNone(opts.save)  # Only parent owns viewer.rrd.


if __name__ == '__main__': unittest.main()
