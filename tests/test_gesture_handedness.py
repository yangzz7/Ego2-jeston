"""Handedness decisions must not flicker, lock forever, or delay skeletons."""
from dataclasses import replace
import json
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

from ego2.gesture.config import DEFAULTS, load_config
from ego2.gesture.dataset import _json
from ego2.gesture.replay import _hands
from ego2.gesture.tracking import HandTracker
from tests.test_gesture import detection


def observed(label="Left", confidence=.95, x=.5):
    return replace(detection(x=x, handedness=label), handedness_score=confidence)


class HandednessTests(TestCase):
    def setUp(self):
        self.tracker = HandTracker({**DEFAULTS, "smoothing_ms": 0})

    def frame(self, stamp, label="Left", confidence=.95, x=.5):
        return self.tracker.update([observed(label, confidence, x)], stamp)[0]

    def confirm(self, label="Left", start=0):
        for dt in (0, .1, .2, .3):
            result = self.frame(start + dt, label)
        self.assertEqual(result.handedness, label)
        return result

    def test_new_hand_is_unknown_until_supported_by_several_frames(self):
        for label in ("Left", "Right"):
            self.tracker = HandTracker(DEFAULTS)
            for stamp in (0, .1, .2):
                hand = self.frame(stamp, label)
                self.assertEqual(hand.handedness, "Unknown")
                self.assertEqual(hand.points.shape, (21, 3))
            self.assertEqual(self.frame(.3, label).handedness, label)

    def test_initial_confirmation_tolerates_weak_frames_without_taking_their_label(self):
        for stamp, label, score in [(0, "Left", .7), (.1, "Left", .7), (.2, "Right", .52),
                                     (.3, "Left", .6), (.4, "Unknown", 0)]:
            self.assertEqual(self.frame(stamp, label, score).handedness, "Unknown")
        self.assertEqual(self.frame(.5, "Left", .7).handedness, "Left")

    def test_initial_evidence_expires_instead_of_accumulating_forever(self):
        self.frame(0)
        for i in range(1, 12):
            self.frame(i * .1, "Left", .55)
        self.assertEqual(self.frame(1.2).handedness, "Unknown")
        self.assertEqual(self.frame(1.4).handedness, "Unknown")
        self.assertEqual(self.frame(1.6).handedness, "Left")

    def test_reliable_opposite_initial_evidence_restarts_confirmation(self):
        self.frame(0, "Right")
        self.frame(.1, "Right")
        for stamp in (.2, .3, .4):
            self.assertEqual(self.frame(stamp, "Left").handedness, "Unknown")
        self.assertEqual(self.frame(.51, "Left").handedness, "Left")

    def test_weak_labels_do_not_create_a_confident_hand(self):
        for i in range(20):
            hand = self.frame(i * .1, "Left" if i % 2 else "Right", .56)
            self.assertEqual(hand.handedness, "Unknown")

    def test_measured_weak_flips_leave_confirmed_label_and_current_points(self):
        previous = self.confirm()
        weak_scores = [.516153, .537132, .557484, .528153, .506786, .563558, .553596, .512365]
        for i, score in enumerate(weak_scores):
            detection = observed("Right", score, .51 + .001 * i)
            original = detection.points.copy()
            hand = self.tracker.update([detection], .4 + .1 * i)[0]
            self.assertEqual((hand.id, hand.handedness), (previous.id, "Left"))
            np.testing.assert_array_equal(hand.points, original)
            np.testing.assert_array_equal(detection.points, original)
            self.assertEqual(detection.handedness, "Right")
            self.assertEqual(detection.handedness_score, score)

    def test_one_high_confidence_error_does_not_flip(self):
        self.confirm()
        self.assertEqual(self.frame(.4, "Right", .99).handedness, "Left")
        for stamp in (.5, .6, .7, .8, .9, 1.):
            self.assertEqual(self.frame(stamp).handedness, "Left")

    def test_sustained_opposite_evidence_can_correct_an_initial_mistake(self):
        self.confirm("Right")
        for stamp in (.4, .5, .6, .7, .8):
            self.assertEqual(self.frame(stamp, "Left").handedness, "Right")
        self.assertEqual(self.frame(.91, "Left").handedness, "Left")

    def test_medium_opposite_evidence_does_not_replace_established_identity(self):
        self.confirm()
        for i in range(20):
            self.assertEqual(self.frame(.4 + .1 * i, "Right", .75).handedness, "Left")

    def test_uncertainty_restarts_pending_switch(self):
        self.confirm()
        for stamp in (.4, .5, .6, .7):
            self.frame(stamp, "Right")
        self.frame(.8, "Right", .52)
        for stamp in (.9, 1., 1.1, 1.2, 1.3):
            self.assertEqual(self.frame(stamp, "Right").handedness, "Left")
        self.assertEqual(self.frame(1.41, "Right").handedness, "Right")

    def test_original_label_restarts_pending_switch(self):
        self.confirm()
        for stamp in (.4, .5, .6, .7):
            self.frame(stamp, "Right")
        self.frame(.8, "Left")
        self.assertEqual(self.frame(.9, "Right").handedness, "Left")

    def test_missing_frame_cannot_complete_pending_initial_confirmation(self):
        self.frame(0)
        self.frame(.1)
        self.assertEqual(self.tracker.update([], .2), [])
        self.assertEqual(self.frame(.3).handedness, "Unknown")
        self.assertEqual(self.frame(.4).handedness, "Unknown")
        self.frame(.5)
        self.assertEqual(self.frame(.61).handedness, "Left")

    def test_brief_occlusion_keeps_identity_but_restarts_pending_switch(self):
        first = self.confirm()
        for stamp in (.4, .5, .6, .7):
            self.frame(stamp, "Right")
        self.assertEqual(self.tracker.update([], .8), [])
        hand = self.frame(.9, "Right")
        self.assertEqual((hand.id, hand.handedness), (first.id, "Left"))
        for stamp in (1., 1.1, 1.2, 1.3):
            self.assertEqual(self.frame(stamp, "Right").handedness, "Left")
        self.assertEqual(self.frame(1.41, "Right").handedness, "Right")

    def test_long_gap_new_track_does_not_inherit_last_hands_label(self):
        first = self.confirm()
        hand = self.frame(.7, "Right")
        self.assertNotEqual(hand.id, first.id)
        self.assertEqual(hand.handedness, "Unknown")
        for stamp in (.8, .9, 1.01):
            hand = self.frame(stamp, "Right")
        self.assertEqual(hand.handedness, "Right")

    def test_low_frame_rate_still_confirms_with_three_real_observations(self):
        self.assertEqual(self.frame(0).handedness, "Unknown")
        self.assertEqual(self.frame(.26).handedness, "Unknown")
        self.assertEqual(self.frame(.52).handedness, "Left")

    def test_two_tracks_and_independent_eyes_do_not_share_label_state(self):
        other_eye = HandTracker(DEFAULTS)
        for i in range(4):
            ds = [observed("Left", x=.25), observed("Right", x=.75)]
            if i % 2:
                ds.reverse()
            hands = self.tracker.update(ds, i * .1)
        labels = {hand.id: hand.handedness for hand in hands}
        self.assertEqual(set(labels.values()), {"Left", "Right"})
        hands = self.tracker.update([observed("Left", .52, .75), observed("Right", .52, .25)], .4)
        self.assertEqual({hand.id: hand.handedness for hand in hands}, labels)
        self.assertEqual(other_eye.update([observed("Right")], 0)[0].handedness, "Unknown")

    def test_unknown_and_invalid_confidence_cannot_override_confirmed_label(self):
        self.confirm()
        for i, (label, score) in enumerate([(None, .99), ("Unknown", .99), ("Right", float('nan')),
                                           ("Right", float('inf')), ("Right", 1.2), ("Right", -.1)]):
            self.assertEqual(self.frame(.4 + .1 * i, label, score).handedness, "Left")

    def test_replay_keeps_stable_label_and_recording_keeps_raw_model_label(self):
        self.confirm()
        hand = self.frame(.4, "Right", .52)
        hand = replace(hand, points=hand.points[:, :2])
        saved = _json(dict(hands=[[hand], []], detections=[[dict(handedness="Right", handedness_score=.52)], []]))
        replayed = _hands(saved, True)[0][0]
        self.assertEqual(replayed.handedness, "Left")
        np.testing.assert_array_equal(replayed.points, hand.points)
        self.assertEqual(json.loads(saved)['detections'][0][0], dict(handedness="Right", handedness_score=.52))


class HandednessConfigTests(TestCase):
    def test_invalid_confidence_and_time_settings_are_rejected(self):
        profile, cameras = load_config()
        bad = [('handedness_confidence', -.1), ('handedness_confidence', float('nan')),
               ('handedness_switch_confidence', 1.1), ('handedness_switch_confidence', .5),
               ('handedness_confirm_ms', -1), ('handedness_initial_window_ms', 200), ('handedness_switch_ms', float('inf')),
               ('handedness_switch_ms', 100)]
        for key, value in bad:
            with self.subTest(key=key, value=value):
                candidate = {**profile, 'gesture': {**profile['gesture'], key: value}}
                with patch('ego2.gesture.config.load_profile', return_value=(candidate, cameras)):
                    with self.assertRaisesRegex(ValueError, 'handedness'):
                        load_config()


if __name__ == '__main__':
    main()
