import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from ego2.common.sources import Pair
import cv2
import numpy as np
from ego2.hand_pose.config import load_config, validate
from ego2.hand_pose.bundle import (Calibrator, synthetic_bundle, fuse, wrist_frame,
                                   save_calibrations, load_bundles, delta)
from ego2.hand_pose.pose import Measurement, Observation, corners, projections
from ego2.hand_pose.detector import Detector
from ego2.hand_pose.demo import DemoSource


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.cameras = load_config()
        self.target = self.cfg['targets'][0]
        self.truth = {int(i): np.array(p) for i, p in synthetic_bundle(self.target).items()}
        self.motion = np.eye(4)
        self.motion[:3, :3] = cv2.Rodrigues(np.array([.1, -.15, .2]))[0] @ np.diag([1., -1., -1.])
        self.motion[:3, 3] = [-.02, .03, .25]

    def calibrated(self, skip=None, bad_edge=None, noise=False):
        calibrator = Calibrator(self.target, self.cfg['bundle_calibration_quality'])
        rng = np.random.default_rng(42)
        ids = self.target['ids']
        for edge, (a, b) in enumerate(zip(ids, ids[1:]+ids[:1])):
            if skip == edge:
                continue
            for k in range(25):
                poses = {}
                for i in (a, b):
                    pose = self.motion @ self.truth[i]
                    if noise:
                        pose[:3, 3] += rng.normal(0, .00015, 3)
                        pose[:3, :3] = pose[:3, :3] @ cv2.Rodrigues(rng.normal(0, .003, 3))[0]
                    if bad_edge == edge and i == b:
                        pose[0, 3] += .10
                    poses[i] = Measurement(pose, .2, 'stereo'), 'ok'
                calibrator.add(edge*3+k*.1, poses)
        return calibrator

    def test_default_real_ids_and_unknown_geometry(self):
        self.assertEqual(self.target['ids'], list(range(94, 101)))
        self.assertEqual(self.cfg['targets'][1]['ids'], list(range(124, 131)))
        self.assertTrue(self.cfg['markers_configured'])
        self.assertNotIn('T_wrist_marker', self.target)

    def test_recovers_irregular_actual_geometry_without_spacing_assumption(self):
        # Unequal angular spacing and an elliptical wrist, no equal-spacing prior.
        ids = self.target['ids']
        angles = np.array([.43, 1.3, 2.27, 3.14, 4.06, 4.94, 5.85])
        for i, a in zip(ids, angles):
            self.truth[i][:3, 3] = [.034*np.sin(a), 0, .025*np.cos(a)]
            self.truth[i][:3, :3] = cv2.Rodrigues(np.array([0., a, .01*np.sin(a)]))[0]
        result = self.calibrated(noise=True).finish()
        estimated = {int(i): np.array(p) for i, p in result['T_wrist_marker'].items()}
        self.assertLess(np.linalg.norm(np.mean([p[:3, 3] for p in estimated.values()], axis=0)), .003)
        for a, b in zip(ids, ids[1:]):
            distance, angle = delta(np.linalg.inv(estimated[a]) @ estimated[b], np.linalg.inv(self.truth[a]) @ self.truth[b])
            self.assertLess(distance, .002)
            self.assertLess(angle, 1.)

    def test_seam_not_observed_cannot_complete_calibration(self):
        with self.assertRaisesRegex(ValueError, 'adjacent pairs'):
            self.calibrated(skip=6).finish()

    def test_inconsistent_loop_is_rejected(self):
        with self.assertRaises(ValueError):
            self.calibrated(bad_edge=6).finish()

    def test_straight_strip_and_wrong_id_order_are_rejected(self):
        strip = {i: p.copy() for i, p in self.truth.items()}
        for k, p in enumerate(strip.values()):
            p[:3, 3] = [k*.03, 0, 0]
        with self.assertRaises(ValueError):
            wrist_frame(strip, self.target['ids'])
        ids = self.target['ids'].copy()
        ids[2], ids[3] = ids[3], ids[2]
        with self.assertRaises(ValueError):
            wrist_frame(self.truth, ids)

    def test_tag_switches_keep_same_wrist_centre_including_seam(self):
        self.target['T_wrist_marker'] = {str(i): p.tolist() for i, p in self.truth.items()}
        for visible in ([94], [95, 96], [97], [99, 100], [100, 94]):
            measurements = {i: (Measurement(self.motion @ p, .2, 'stereo'), 'ok') if i in visible
                            else (None, 'not_seen') for i, p in self.truth.items()}
            result, reason = fuse(self.target, measurements, self.cfg['bundle_fusion'])
            self.assertEqual(reason, 'ok')
            np.testing.assert_allclose(result.T_head_marker, self.motion, atol=1e-8)
            self.assertEqual(set(result.marker_ids), set(visible))

    def test_deformation_rejected_instead_of_averaging_conflicting_hands(self):
        self.target['T_wrist_marker'] = {str(i): p.tolist() for i, p in self.truth.items()}
        measurements = {i: (None, 'not_seen') for i in self.truth}
        for i in (94, 100):
            pose = self.motion @ self.truth[i]
            if i == 100:
                pose[0, 3] += .04
            measurements[i] = Measurement(pose, .2, 'stereo'), 'ok'
        self.assertEqual(fuse(self.target, measurements, self.cfg['bundle_fusion'])[1], 'bundle_geometry_changed')

    def test_camera_signature_size_and_ids_checked_and_other_hand_preserved(self):
        entry = self.calibrated().finish()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wrists.json'
            cfg = copy.deepcopy(self.cfg)
            cfg['targets'] = [cfg['targets'][0]]
            cfg['bundle_calibration'] = str(path)
            save_calibrations(path, cfg, {'left': entry})
            save_calibrations(path, cfg, {'right': {'sentinel': True}})
            self.assertTrue(json.loads(path.read_text())['targets']['right']['sentinel'])
            load_bundles(cfg)
            cfg['targets'][0]['size_m'] = .03
            with self.assertRaises(ValueError):
                load_bundles(cfg)
            cfg['targets'][0]['size_m'] = .0225
            cfg['border_bits'] = 1
            with self.assertRaises(ValueError):
                load_bundles(cfg)

    def test_overlapping_left_right_ids_rejected(self):
        self.cfg['targets'][1]['ids'][0] = 94
        with self.assertRaises(ValueError):
            validate(self.cfg)

    def test_missing_wrist_calibration_does_not_open_camera(self):
        from ego2.hand_pose.__main__ import main
        self.cfg['bundle_calibration'] = '/tmp/no-such-wrist-test-calibration.json'
        with patch('ego2.hand_pose.__main__.load_config', return_value=(self.cfg, self.cameras)), patch('ego2.hand_pose.app.run') as run:
            self.assertEqual(main([]), 1)
            run.assert_not_called()

    def test_joint_corners_recover_pose_and_fisheye_demo_detects_real_tags(self):
        source = DemoSource(self.cfg, self.cameras)
        source.open()
        detector = Detector(self.cfg, self.cameras)
        pair = source.read(threading.Event())
        result = detector.process(pair.images)
        valid = 0
        for name, (measurement, _) in result.items():
            if measurement is not None:
                valid += 1
                distance, angle = delta(measurement.T_head_marker, source.truth[name])
                self.assertLess(distance, .006)
                self.assertLess(angle, 12.)
                self.assertTrue(measurement.marker_ids)
        self.assertGreaterEqual(valid, 1, detector.last_seen)

    def test_manual_palm_rotation_keeps_centre_without_recapture(self):
        entry = self.calibrated().finish()
        with tempfile.TemporaryDirectory() as directory:
            cfg = copy.deepcopy(self.cfg)
            cfg['targets'] = [cfg['targets'][0]]
            cfg['bundle_calibration'] = str(Path(directory)/'wrists.json')
            save_calibrations(cfg['bundle_calibration'], cfg, {'left': entry})
            cfg['targets'][0]['palm_offset_deg'] = 12
            load_bundles(cfg)
            correction = np.eye(4)
            correction[:3, :3] = cv2.Rodrigues(np.array([0., -np.deg2rad(12), 0.]))[0]
            for i in self.target['ids']:
                np.testing.assert_allclose(cfg['targets'][0]['T_wrist_marker'][str(i)],
                                           correction @ np.array(entry['T_wrist_marker'][str(i)]), atol=1e-8)

    def test_calibration_command_writes_real_geometry_and_keeps_other_hand(self):
        from ego2.hand_pose.__main__ import main
        ids = self.target['ids']
        pairs, measured = [], []
        for edge, (a, b) in enumerate(zip(ids, ids[1:]+ids[:1])):
            for k in range(25):
                stamp = edge*3+k*.1
                pairs.append(Pair((), stamp, stamp, len(pairs)))
                measured.append({i: (Measurement(self.motion @ self.truth[i], .2, 'stereo'), 'ok')
                                 if i in (a, b) else (None, 'not_seen') for i in ids})
        iterator = iter(pairs)
        source = SimpleNamespace(kind='images', open=lambda: None, close=lambda: None,
                                 read=lambda stop: next(iterator, None))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wrists.json'
            save_calibrations(path, self.cfg, {'right': {'sentinel': True}})
            session = Path(directory)/'capture'
            with patch('ego2.hand_pose.app.Source', return_value=source), patch.object(Detector, 'measure', side_effect=measured):
                self.assertEqual(main(['calibrate', '--hand', 'left', '--no-viz', '--frames', str(len(pairs)),
                                       '--session', str(session), '--bundle-calibration', str(path)]), 0)
            document = json.loads(path.read_text())
            self.assertEqual(document['targets']['left']['ids'], ids)
            self.assertTrue(document['targets']['right']['sentinel'])
            self.assertTrue(json.loads((session/'session.json').read_text())['finished'])
            self.assertEqual(len((session/'marker_samples.jsonl').read_text().splitlines()), len(pairs))
            # A failed subsequent capture must not replace either hand's calibration.
            original = path.read_bytes()
            empty = SimpleNamespace(kind='images', open=lambda: None, close=lambda: None,
                                    read=lambda stop: Pair((), 0, 0, 0))
            with patch('ego2.hand_pose.app.Source', return_value=empty), patch.object(Detector, 'measure', return_value={i: (None, 'not_seen') for i in ids}):
                self.assertEqual(main(['calibrate', '--hand', 'left', '--no-viz', '--frames', '1',
                                       '--session', str(Path(directory)/'failed'), '--bundle-calibration', str(path)]), 1)
            self.assertEqual(path.read_bytes(), original)

    def test_synthetic_mode_never_saves_real_calibration(self):
        from ego2.hand_pose.__main__ import main
        with patch('ego2.hand_pose.app.run') as run:
            self.assertEqual(main(['calibrate', '--demo']), 1)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
