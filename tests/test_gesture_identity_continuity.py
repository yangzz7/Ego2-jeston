"""Real tracker-to-3D regressions for weak labels, gaps and same-frame timing."""
import copy
from dataclasses import replace
import json
import pickle
import queue
import unittest
from unittest.mock import MagicMock

import numpy as np
from ego2.gesture.config import DEFAULTS
from ego2.gesture.dataset import _json
from ego2.gesture.replay import _hands
from ego2.gesture.recognizer import Detection
from ego2.gesture.tracking import HandTracker
from ego2.gesture.app import Pipeline, FrameClock
from ego2.common.sources import Pair
from ego2.common.mahony import increment, rotation
from ego2.multi_vio.hand_shape import candidate, pack_observations, HandSelector
from ego2.multi_vio.body_preview import BodyPreview, load_settings
from ego2.multi_vio.mode_viewer import BatchRecording, GestureSink
from tests.test_body_preview import observation, pose_packet
from tests import test_body_preview as body_fixtures


def detected(side='Left', score=.95, raw_side=None, dx=0., bent=False):
    item = observation(side, score=score, bent=bent)
    item['points'][:,0] += dx
    return Detection(item['points'], raw_side or side, score, 'Open_Palm', .9, item['world_points'])


def frame(tracker, stamp, detection):
    hands = tracker.update([detection], stamp)
    item = dict(points=detection.points, world_points=detection.world_points,
                handedness=detection.handedness, handedness_score=detection.handedness_score,
                track_id=hands[0].id, visible=np.ones(21,bool))
    packed = pack_observations([[item]], [hands], ['cam0'])
    return hands[0], packed[0], item


def confirm(tracker, side='Left', score=.95, dx=0.):
    for t in (0.,.1,.3):
        hand,item,raw = frame(tracker,t,detected(side,score,dx=dx))
    return hand,item,raw


class IdentityContinuityTests(unittest.TestCase):
    def tracker(self):
        return HandTracker({**DEFAULTS,'smoothing_ms':0.})

    def test_confirmed_moderate_side_is_shared_with_3d(self):
        for side in ('Left','Right'):
            h,p,raw=confirm(self.tracker(),side,.70)
            self.assertEqual(h.handedness,side)
            self.assertEqual(candidate('chest',.3,p).side,side)
            self.assertEqual(p['handedness_score'],.70)

    def test_weak_flip_keeps_identity_and_current_geometry_without_changing_raw(self):
        for side,opposite in [('Left','Right'),('Right','Left')]:
            tr=self.tracker();first,_,_=confirm(tr,side)
            d=detected(side,.52,opposite,dx=.02,bent=True)
            original=copy.deepcopy(d)
            h,p,raw=frame(tr,.4,d)
            self.assertEqual((h.id,h.handedness),(first.id,side))
            self.assertEqual(p['identity']['status'],'held')
            self.assertEqual(candidate('chest',.4,p).stamp,.4)
            np.testing.assert_array_equal(h.points,d.points)
            np.testing.assert_array_equal(p['world_points'],d.world_points.astype(np.float32))
            self.assertEqual((raw['handedness'],raw['handedness_score']),(opposite,.52))
            np.testing.assert_array_equal(original.points,d.points)
            np.testing.assert_array_equal(original.world_points,d.world_points)
            p['world_points'][:]=0
            np.testing.assert_array_equal(original.world_points,d.world_points)

    def test_weak_initial_evidence_never_creates_a_confirmed_hand(self):
        tr=self.tracker()
        for i in range(12):
            h,p,_=frame(tr,i*.1,detected(score=.56))
            self.assertEqual(h.handedness,'Unknown')
            self.assertEqual(p['rejection_reason'],'side_unconfirmed')

    def test_weak_hold_expires_and_current_support_recovers(self):
        tr=self.tracker();confirm(tr)
        for t in (.4,.5,.6,.7,.8,.9):
            _,p,_=frame(tr,t,detected(score=.52,raw_side='Right'))
            candidate('head',t,p)
        _,p,_=frame(tr,1.,detected(score=.52,raw_side='Right'))
        self.assertEqual(p['rejection_reason'],'side_evidence_expired')
        _,p,_=frame(tr,1.1,detected(score=.7))
        self.assertEqual(candidate('head',1.1,p).side,'Left')

    def test_strong_opposite_is_withheld_until_reconfirmed(self):
        tr=self.tracker();confirm(tr)
        for t in (.4,.5,.6,.7,.8):
            h,p,_=frame(tr,t,detected('Right',.95))
            self.assertEqual(h.handedness,'Left')
            self.assertEqual(p['rejection_reason'],'side_conflict')
        h,p,_=frame(tr,.91,detected('Right',.95))
        self.assertEqual(candidate('head',.91,p).side,'Right')

    def test_medium_opposite_and_invalid_score_are_not_accepted_via_stable_label(self):
        for score,reason in [(.75,'side_conflict'),(float('nan'),'invalid_side_evidence'),
                              (float('inf'),'invalid_side_evidence'),(-.1,'invalid_side_evidence'),
                              (1.1,'invalid_side_evidence')]:
            tr=self.tracker();confirm(tr)
            _,p,_=frame(tr,.4,detected(score=score,raw_side='Right'))
            self.assertEqual(p['rejection_reason'],reason)

    def test_strict_geometry_gates_remain_active_with_held_identity(self):
        for change,message in [('crop','Cropped hand'),('small','Hand too small'),
                               ('basis','Degenerate palm basis'),('bone','Implausible bone length')]:
            tr=self.tracker();confirm(tr)
            _,p,_=frame(tr,.4,detected(score=.52,raw_side='Right'))
            if change=='crop':p['visible'][5]=False
            if change=='small':p['points'][:,:2]=.5+(p['points'][:,:2]-.5)*.01
            if change=='basis':p['world_points'][17]=p['world_points'][5]
            if change=='bone':p['world_points'][8]=p['world_points'][7]
            with self.assertRaisesRegex(ValueError,message):candidate('head',.4,p)

    def test_legacy_replay_stays_strict_and_new_replay_preserves_raw_labels(self):
        tr=self.tracker();confirm(tr)
        h,p,raw=frame(tr,.4,detected(score=.52,raw_side='Right'))
        h=replace(h,points=h.points[:,:2])  # Pipeline backprojection for display/replay.
        stored=_json(dict(hands=[[h],[]],detections=[[raw],[]]))
        restored=_hands(stored,True)[0][0]
        self.assertEqual(restored.identity,h.identity)
        self.assertEqual(json.loads(stored)['detections'][0][0]['handedness'],'Right')
        data=json.loads(stored);data['hands'][0][0].pop('identity')
        self.assertIsNone(_hands(json.dumps(data),True)[0][0].identity)
        old=observation(score=.7)
        with self.assertRaisesRegex(ValueError,'Uncertain side'):candidate('head',.4,old)

    def test_confirmed_gap_reuses_id_but_emits_nothing_without_observation(self):
        for gap in (.4,.55,.64):
            tr=self.tracker();first,_,_=confirm(tr)
            self.assertEqual(tr.update([],.4),[])
            h,p,_=frame(tr,.3+gap,detected(dx=.015,bent=True))
            self.assertEqual((h.id,h.handedness),(first.id,'Left'))
            candidate('head',.3+gap,p)
            np.testing.assert_array_equal(h.points,detected(dx=.015,bent=True).points)
            self.assertEqual(tr.stats['track_reacquired'],1)

    def test_long_gap_and_unconfirmed_track_do_not_inherit_identity(self):
        tr=self.tracker();first,_,_=confirm(tr)
        h,p,_=frame(tr,.96,detected())
        self.assertNotEqual(h.id,first.id)
        self.assertEqual(p['rejection_reason'],'side_unconfirmed')
        tr=self.tracker();first,_,_=frame(tr,0.,detected())
        h,_,_=frame(tr,.31,detected())
        self.assertNotEqual(h.id,first.id)

    def test_recovery_requires_same_side_visible_palm_and_local_position(self):
        for kind in ('opposite','weak','far','scale','palm_shape','invisible'):
            tr=self.tracker();first,_,_=confirm(tr);d=detected()
            visible=np.ones(21,bool)
            if kind=='opposite':d=detected('Right')
            if kind=='weak':d.handedness_score=.52
            if kind=='far':d.points[:,0]+=.13
            if kind=='scale':
                center=d.points[[0,5,9,13,17],:2].mean(0)
                d.points[:,:2]=center+2*(d.points[:,:2]-center)
            if kind=='palm_shape':d.points[[5,17]]=d.points[[17,5]]
            if kind=='invisible':visible[5]=False
            h=tr.update([d],.8,[visible])[0]
            self.assertNotEqual(h.id,first.id,kind)
            self.assertEqual(h.handedness,'Unknown',kind)

    def test_ambiguous_recovery_never_picks_the_later_candidate_by_loop_order(self):
        tr=self.tracker();first,_,_=confirm(tr)
        hands=tr.update([detected(dx=-.005),detected(dx=.005)],.8)
        self.assertTrue(all(h.id!=first.id and h.handedness=='Unknown' for h in hands))

    def test_two_hands_reordered_after_gap_keep_independent_ids(self):
        tr=self.tracker()
        for t in (0.,.1,.3):
            old=tr.update([detected(dx=-.2),detected('Right',dx=.2)],t)
        new=tr.update([detected('Right',dx=.21),detected(dx=-.19)],.8)
        self.assertEqual([h.id for h in new],[old[1].id,old[0].id])
        self.assertEqual([h.handedness for h in new],['Right','Left'])
        other=self.tracker();h=other.update([detected()],.8)[0]
        self.assertEqual(h.handedness,'Unknown')

    def test_weak_flip_and_brief_gap_do_not_force_source_handover(self):
        tr=self.tracker();_,p,_=confirm(tr);s=HandSelector()
        s.update('chest',.3,[p]);source=s.choose('Left',.31).source
        _,p,_=frame(tr,.4,detected(score=.52,raw_side='Right'))
        s.update('chest',.4,[p]);self.assertEqual(s.choose('Left',.41).source,source)
        self.assertEqual(tr.update([],.5),[])
        s.update('chest',.5,[]);self.assertIsNone(s.choose('Left',.51))
        _,p,_=frame(tr,.9,detected())
        s.update('chest',.9,[p]);chosen=s.choose('Left',.91)
        self.assertEqual((chosen.source,chosen.stamp),(source,.9))
        self.assertEqual(s.stats['switches'],0)

    def test_held_identity_uses_hand_camera_pose_at_visual_time_not_latest(self):
        tr=self.tracker();confirm(tr)
        _,p,_=frame(tr,.4,detected(score=.52,raw_side='Right',bent=True))
        rec=MagicMock();body=BodyPreview(rec,['chest','left_hand'],settings=load_settings())
        body.clocks['chest']=0.
        imu=pose_packet('left_hand',start=1.,n=101)
        for row in imu['imu']:row[2:]=increment([0,0,row[0]-1.])
        body.consume_imu(imu,1.51)
        body.consume_visual(dict(role='chest',received=1.4,image_device_time=1.4,observations=[p]),1.51)
        body.tick(1.51)
        self.assertEqual(body.last_render['left_hand']['stamp'],1.4)
        actual=body_fixtures.MountTests.logged_transform(rec,'/body/left_hand/camera')
        np.testing.assert_allclose(actual,rotation(increment([0,0,.4])),atol=1e-6)
        self.assertFalse(body.last_render['left_hand']['grey'])

    def test_tracking_diagnostics_keep_capture_time_and_survive_dropped_preview_packets(self):
        tr=self.tracker();confirm(tr)
        tr.update([],.4);_,p,_=frame(tr,.8,detected())
        diag=MagicMock();body=BodyPreview(MagicMock(),['chest'],settings=load_settings(),diagnostics=diag)
        body.clocks['chest']=10.
        packet=dict(role='chest',received=10.8,image_device_time=.8,observations=[p],
                    tracking={'cam0':dict(events=tr.events,stats=dict(tr.stats))})
        body.consume_visual(packet,10.9)
        self.assertEqual(body.summary()['tracking']['chest/cam0']['track_created'],1)
        self.assertEqual(body.summary()['tracking']['chest/cam0']['track_reacquired'],1)
        events=[c.args[0] for c in diag.emit.call_args_list if c.args[0]['kind']=='tracking']
        self.assertEqual(events[0]['stamp'],10.8)
        self.assertEqual(events[0]['reason'],'track_reacquired')
        body.consume_visual(packet,10.91)
        self.assertEqual(len([c for c in diag.emit.call_args_list if c.args[0]['kind']=='tracking']),1)

    def test_pipeline_transports_identity_and_diagnostics_with_the_same_stereo_frame(self):
        pipeline=Pipeline.__new__(Pipeline);pipeline.clock=FrameClock()
        pipeline.trackers=[self.tracker(),self.tracker()]
        pipeline.rectifiers=[];pipeline.recognizers=[]
        for eye in range(2):
            rect=MagicMock();rect.prepare.side_effect=lambda image:image
            rect.backproject.side_effect=lambda p:(p[:,:2]*100,np.ones(21,bool))
            pipeline.rectifiers.append(rect)
            recognizer=MagicMock();recognizer.detect.return_value=[detected()]
            pipeline.recognizers.append(recognizer)
        for stamp in (0.,.1,.3):
            pair=Pair(tuple(np.full((100,100,3),30+eye*50,np.uint8) for eye in range(2)),
                      stamp,10.+stamp,int(stamp*10),{'image_timestamp_seconds':stamp})
            capture,hands=pipeline.process(pair)
        for r in pipeline.recognizers:r.detect.return_value=[detected(score=.52,raw_side='Right')]
        pair=replace(pair,stamp=.4,received=10.4,sequence=4,metadata={'image_timestamp_seconds':.4})
        capture,hands=pipeline.process(pair)
        q=queue.Queue();rec=BatchRecording(q,'chest',10.,eye_names=('cam0','cam1'))
        sink=GestureSink({'viewer':{}},rec)
        sink.set_tracking_diagnostics(pipeline.tracking_diagnostics)
        sink.log_observations(pair,capture,hands,pipeline.observations)
        batch=pickle.loads(q.get_nowait());visual=batch['hand_visual']
        self.assertAlmostEqual(visual['received'],.4)
        self.assertEqual(visual['image_device_time'],.4)
        self.assertEqual(len(visual['observations']),2)
        for item in visual['observations']:
            self.assertEqual(candidate('chest',.4,item).side,'Left')
            self.assertEqual(item['raw_handedness'],'Right')
            self.assertEqual(item['identity']['status'],'held')
        for eye in ('cam0','cam1'):
            self.assertEqual(visual['tracking'][eye]['stats']['observed'],4)
            self.assertEqual(visual['tracking'][eye]['stats']['side_held'],1)
        for path,items,kwargs,times in batch['logs']:
            self.assertAlmostEqual(times['capture_time']['duration'],.4)
        self.assertTrue(all(row['handedness']=='Right' and row['handedness_score']==.52
                            for rows in pipeline.observations for row in rows))


if __name__=='__main__':unittest.main()
