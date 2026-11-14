import unittest
import numpy as np
from ego2.hand_pose.config import load_config
from ego2.hand_pose.diagnostics import ObservationDiagnostics
from ego2.hand_pose.pose import Measurement


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        cfg, _ = load_config()
        cfg['targets'] = cfg['targets'][:1]
        self.monitor = ObservationDiagnostics(cfg)
        self.empty = {i: (None, 'not_seen') for i in range(94, 101)}

    def test_no_detection_is_different_from_stereo_failure(self):
        self.monitor.update(0, [], self.empty)
        self.assertEqual(self.monitor.summary()['targets']['left']['state'], 'no_target_decoded')
        measured = {**self.empty, 94: (None, 'need_both_cameras')}
        self.monitor.update(1, [{'camera': 'cam0', 'id': 94}], measured)
        result = self.monitor.summary()['targets']['left']
        self.assertEqual(result['state'], 'no_valid_pose')
        self.assertEqual(result['rejected_pose_counts'], {'need_both_cameras': 1})
        self.assertIn('两眼', result['hint'])

    def test_foreign_wrist_does_not_count_as_this_wrist(self):
        self.monitor.update(0, [{'camera': 'cam0', 'id': 124}], self.empty)
        result = self.monitor.summary()['targets']['left']
        self.assertEqual(result['decoded_ids'], [])
        self.assertEqual(result['state'], 'no_target_decoded')

    def test_zero_valid_observations_fail_early_without_touching_thresholds(self):
        self.monitor.update(0, [], self.empty)
        self.monitor.update(14.9, [], self.empty)
        self.assertFalse(self.monitor.stalled())
        self.monitor.update(15, [], self.empty)
        self.assertTrue(self.monitor.stalled())

    def test_single_tag_and_coview_pairs_have_different_guidance(self):
        m = Measurement(np.eye(4), .2, 'stereo')
        seen = [{'camera': 'cam0', 'id': 94}, {'camera': 'cam1', 'id': 94}]
        self.monitor.update(0, seen, {**self.empty, 94: (m, 'ok')})
        self.assertEqual(self.monitor.summary()['targets']['left']['state'], 'no_coview_pair')
        self.monitor.update(16, seen+[{'camera': 'cam0', 'id': 95}], {**self.empty, 94: (m, 'ok'), 95: (m, 'ok')})
        self.assertFalse(self.monitor.stalled())
        self.assertEqual(self.monitor.summary()['targets']['left']['state'], 'collecting_pairs')
