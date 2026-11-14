"""Behavioral checks for source loss, causal articulation and audit events."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace
import numpy as np
from ego2.multi_vio.hand_shape import HandSelector,HandShapeFilter,pack_observations,retarget,palm_local,template,EDGES,PALM
from ego2.multi_vio.hand_diagnostics import HandDiagnostics
from ego2.multi_vio.body_preview import BodyPreview,DEFAULTS
from ego2.common.mahony import increment,rotation
from tests.test_body_preview import observation,pose_packet
from tests import test_body_preview as fixtures


def shape(angle=0., side='Left'):
    p=template(side)
    for root in (5,9,13,17):
        for j in range(1,4):
            length=np.linalg.norm(template(side)[root+j]-template(side)[root+j-1])
            p[root+j]=p[root+j-1]+[0,length*math.cos(angle*j),length*math.sin(angle*j)]
    return retarget(palm_local(p,side),side)


class SelectionTests(unittest.TestCase):
    def start(self):
        s=HandSelector();s.update('head',1.,[observation()]);s.choose('Left',1.01)
        s.update('chest',1.,[observation(score=.99)])
        return s

    def test_one_frame_loss_and_return_does_not_switch_or_reuse_missing_observation(self):
        s=self.start();s.update('head',1.1,[])
        self.assertIsNone(s.choose('Left',1.11))
        s.update('head',1.2,[observation()])
        self.assertEqual(s.choose('Left',1.21).source[0],'head')
        self.assertEqual(s.stats['switches'],0)

    def test_ticks_do_not_confirm_a_single_challenger_sample(self):
        s=self.start();s.update('head',1.1,[])
        for t in np.arange(1.11,1.6,.01):self.assertIsNone(s.choose('Left',t))
        self.assertEqual(s.stats['switches'],0)

    def test_sustained_replacement_switches_once_and_preserves_source_timestamp(self):
        s=self.start();s.update('head',1.1,[])
        for t in (1.1,1.2,1.3):
            s.update('chest',t,[observation()]);self.assertIsNone(s.choose('Left',t+.01))
        s.update('chest',1.4,[observation()]);c=s.choose('Left',1.41)
        self.assertEqual(c.source,('chest','cam0',1));self.assertEqual(c.stamp,1.4)
        s.choose('Left',1.42);self.assertEqual(s.stats['switches'],1)

    def test_alternating_or_disappearing_challengers_cannot_confirm_during_grace(self):
        s=self.start();s.update('head',1.1,[])
        for i,t in enumerate(np.arange(1.1,1.7,.1)):
            s.update('chest',t,[observation(eye='cam0' if i%2 else 'cam1')])
            self.assertIsNone(s.choose('Left',t+.01))
        self.assertEqual(s.stats['switches'],0)

    def test_asynchronous_good_cameras_do_not_starve_replacement_confirmation(self):
        s=self.start();s.update('head',1.1,[])
        for t in (1.1,1.2,1.3,1.4):
            s.update('chest',t,[observation()]);s.choose('Left',t+.01)
            s.update('right_hand',t+.025,[observation(score=.96)])
            chosen=s.choose('Left',t+.03)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.source[0],'chest')
        self.assertEqual(s.stats['switches'],1)

    def test_rejected_frames_break_challenger_evidence(self):
        s=self.start();s.update('head',1.1,[])
        s.update('chest',1.1,[observation()]);s.choose('Left',1.11)
        s.update('chest',1.2,[observation(score=.5)]);s.choose('Left',1.21)
        s.update('chest',1.4,[observation()]);self.assertIsNone(s.choose('Left',1.41))

    def test_nonmonotonic_packet_cannot_remove_newer_candidates(self):
        s=self.start();s.update('head',.9,[])
        self.assertEqual(s.candidates[('head','cam0',1)].stamp,1.)
        self.assertEqual(s.reasons['nonmonotonic_visual'],1)

    def test_older_candidate_never_moves_selected_time_backwards(self):
        s=self.start();s.update('head',1.3,[observation()]);s.choose('Left',1.31)
        s.update('head',1.4,[])
        s.update('chest',1.2,[observation()]);self.assertIsNone(s.choose('Left',1.41))

    def test_every_filter_has_reason_and_eye_track_provenance(self):
        events=[];s=HandSelector(on_event=events.append)
        bad=observation();bad['visible'][5]=False
        s.update('head',1.,[observation(score=.5),dict(observation(),eye='cam1',rejection_reason='side_unconfirmed'),bad])
        rows=[e for e in events if e['kind']=='filter']
        self.assertEqual(len(rows),3)
        self.assertEqual({e['reason'] for e in rows},{'side_score_low','side_unconfirmed','cropped_hand'})
        self.assertTrue(all(e['source'][0]=='head' and len(e['source'])==3 for e in rows))
        self.assertEqual(sum(v['rejected'] for v in s.sources.values()),3)

    def test_preselection_side_filters_are_reported_without_mutating_model(self):
        item=observation();item.update(track_id=7,handedness='Right');original=copy.deepcopy(item)
        for side,reason in [('Unknown','side_unconfirmed'),('Left','side_conflict')]:
            out=pack_observations([[item]],[[SimpleNamespace(id=7,handedness=side)]],['cam0'])
            self.assertEqual(out[0]['rejection_reason'],reason)
            self.assertNotIn('world_points',out[0])
        np.testing.assert_array_equal(item['world_points'],original['world_points'])


class FilterTests(unittest.TestCase):
    def test_static_noise_reduced_without_bone_shrink_or_palm_motion(self):
        f=HandShapeFilter();raws=[];filtered=[]
        for i in range(90):
            angle=.02*(-1)**i;p=shape(angle);before=p.copy()
            out=f.update(p,'Left',1+i*.1,('head','cam0',1))
            np.testing.assert_array_equal(p,before)
            for a,b in EDGES:self.assertAlmostEqual(np.linalg.norm(out[b]-out[a]),np.linalg.norm(p[b]-p[a]),places=10)
            np.testing.assert_allclose(out[PALM].mean(axis=0),0,atol=1e-12)
            raws.append(p[12,2]);filtered.append(out[12,2])
        self.assertLess(np.std(filtered[10:]),.8*np.std(raws[10:]))

    def test_fast_finger_movement_follows_current_sample_without_frame_queue(self):
        for hz in (9.,15.,30.):
            f=HandShapeFilter();a=shape(0);b=shape(.6)
            f.update(a,'Left',1.,('head','cam0',1))
            out=f.update(b,'Left',1+1/hz,('head','cam0',1))
            # Current step immediately affects the skeleton; near target within
            # two inference intervals, never an extra queued display frame.
            self.assertLess(np.linalg.norm(out-b),.7*np.linalg.norm(a-b))
            out=f.update(b,'Left',1+2/hz,('head','cam0',1))
            self.assertLess(np.linalg.norm(out-b),.3*np.linalg.norm(a-b))

    def test_source_epoch_and_long_gap_reset_without_cross_view_blending(self):
        for change in ('source','epoch','gap'):
            f=HandShapeFilter();f.update(shape(),'Left',1.,('head','cam0',1))
            p=shape(.5)
            result=f.update(p,'Left',1.5 if change=='gap' else 1.1,
                ('chest','cam0',1) if change=='source' else ('head','cam0',1),1 if change=='epoch' else 0)
            np.testing.assert_allclose(result,p,atol=1e-12)

    def test_left_and_right_straight_finger_axes_unchanged(self):
        for side in ('Left','Right'):
            p=shape(side=side);out=HandShapeFilter().update(p,side,1.,('head','cam0',1))
            np.testing.assert_allclose(out,p,atol=1e-12)

    def test_future_inputs_cannot_affect_previous_output(self):
        f=HandShapeFilter();out=f.update(shape(),'Left',1.,('head','cam0',1));saved=out.copy()
        f.update(shape(.7),'Left',1.1,('head','cam0',1))
        np.testing.assert_array_equal(out,saved)


class IntegrationTests(unittest.TestCase):
    def test_filtered_shape_and_rotation_keep_visual_timestamp(self):
        rec=MagicMock();b=BodyPreview(rec,['head','left_hand'],settings=copy.deepcopy(DEFAULTS))
        for role in ('head','left_hand'):
            p=pose_packet(role,start=1,n=61)
            if role=='left_hand':
                for row in p['imu']:row[2:]=increment([0,0,(row[0]-1)*2])
            b.consume_imu(p,1.31)
        for stamp,bent in [(1.05,False),(1.15,True)]:
            b.consume_visual(dict(role='head',received=stamp,image_device_time=stamp,observations=[observation(bent=bent)]),stamp+.12)
            b.tick(stamp+.12)
            self.assertEqual(b.last_render['left_hand']['stamp'],stamp)
            matrix=fixtures.MountTests.logged_transform(rec,'/body/left_hand/model')
            np.testing.assert_allclose(matrix,rotation(increment([0,0,(stamp-1)*2])),atol=1e-6)
        updates=b.shape_filters['left_hand'].stats['updates']
        b.tick(1.28);self.assertEqual(b.shape_filters['left_hand'].stats['updates'],updates)
        b.consume_visual(dict(role='head',received=1.25,image_device_time=1.25,observations=[]),1.3)
        b.tick(1.3);self.assertEqual(b.last_render['left_hand']['stamp'],1.15)
        b.tick(1.8);self.assertTrue(b.last_render['left_hand']['grey'])
        b.tick(3);self.assertTrue(b.last_render['left_hand']['hidden'])

    def test_event_log_rotates_and_shutdown_flushes(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=HandDiagnostics(tmp,max_bytes=2000,backups=2)
            for i in range(80):log.emit(dict(kind='filter',reason='cropped_hand',source=['head','cam0',i],stamp=i*.1))
            log.close({'switches':2})
            paths=list(Path(tmp).glob('events.jsonl*'))
            self.assertLessEqual(len(paths),3);self.assertGreater(len(paths),1)
            self.assertEqual(log.stats['dropped'],0);self.assertEqual(log.stats['errors'],0)
            self.assertFalse(log.thread.is_alive())
            rows=[json.loads(line) for p in paths for line in p.read_text().splitlines()]
            self.assertTrue(any(r['reason']=='end' and r['summary']['switches']==2 for r in rows))
            self.assertTrue(all(r['session']==log.session for r in rows))

    def test_pose_wait_logged_once_per_observation_not_per_viewer_tick(self):
        events=[];diag=SimpleNamespace(emit=events.append)
        b=BodyPreview(MagicMock(),['head','left_hand'],settings=copy.deepcopy(DEFAULTS),diagnostics=diag)
        b.consume_imu(pose_packet('head'),1.11)
        b.consume_visual(dict(role='head',received=1.05,image_device_time=1.05,observations=[observation()]),1.15)
        for now in (1.15,1.2,1.25):b.tick(now)
        self.assertEqual(sum(e.get('reason')=='matching_pose_unavailable' for e in events),1)

if __name__=='__main__':unittest.main()
