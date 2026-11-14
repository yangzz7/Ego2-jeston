"""Regression for delayed observations and Rerun capture-time recovery."""
import copy
import math
import unittest
from types import SimpleNamespace
import numpy as np
import rerun as rr
from ego2.common.mahony import increment, rotation
from ego2.multi_vio.body_preview import BodyPreview, load_settings
from ego2.multi_vio.hand_shape import HandSelector
from tests.test_body_preview import observation, pose_packet


class TimelineRecording:
    """Collect actual SDK payloads; query each component by timeline, not call order."""
    def __init__(self): self.stamp=0.;self.rows=[]
    def reset_time(self): self.stamp=0.
    def set_time(self, _name, *, duration): self.stamp=float(duration)
    def log(self,path,payload,static=False): self.rows.append((self.stamp,path,payload,static))
    def latest(self,path,field,at=math.inf):
        rows=[];clear=(-math.inf,-1)
        for order,(t,p,obj,static) in enumerate(self.rows):
            if static or t>at:continue
            if isinstance(obj,rr.Clear) and (path==p or path.startswith(p+'/')):
                clear=max(clear,(t,order))
            value=getattr(obj,field,None)
            if p==path and value is not None:rows.append((t,order,value.as_arrow_array().to_pylist()))
        rows=[r for r in rows if r[:2]>clear]
        return max(rows,key=lambda r:r[:2])[2] if rows else None
    def latest_time(self,path,field):
        return max(t for t,p,a,_ in self.rows if p==path and getattr(a,field,None) is not None)


def colors(rgb):return rr.Points3D([[0,0,0]],colors=rgb).colors.as_arrow_array().to_pylist()


class DelayedSelectionTests(unittest.TestCase):
    def test_valid_delayed_replacement_does_not_pin_dead_source(self):
        for delay in (.3,.4,.5,.6):
            for side in ('Left','Right'):
                with self.subTest(delay=delay,side=side):
                    s=HandSelector();s.update('head',1,[observation(side)])
                    s.choose(side,1+delay);s.update('head',1.1,[])
                    got=[]
                    for i in range(1,30):
                        stamp=1.1+i*.1
                        s.update('chest',stamp,[observation(side)])
                        c=s.choose(side,stamp+delay)
                        if c:got.append(c.stamp)
                    self.assertGreater(len(got),20)
                    self.assertLessEqual(got[0],1.9)
                    self.assertEqual(s.stats['switches'],1)
                    self.assertEqual(s.selected[side],('chest','cam0',1))
                    self.assertTrue(all(b>a for a,b in zip(got,got[1:])))

    def test_delayed_churn_recovers_but_does_not_accept_expired_results(self):
        s=HandSelector();s.update('head',1,[observation()]);s.choose('Left',1.4)
        s.update('head',1.1,[]);out=[]
        for i in range(1,50):
            stamp=1.1+i*.1
            s.update('chest',stamp,[observation(track=i,score=.5 if i%2 else .95)])
            c=s.choose('Left',stamp+.4)
            if c:out.append(c.stamp)
        self.assertGreater(len(out),4)
        self.assertLessEqual(max(np.diff(out)),.9)
        self.assertIsNone(s.choose('Left',7))
        s.update('chest',7,[observation(track=100)])
        self.assertIsNone(s.choose('Left',7.66))
        self.assertIsNone(s.choose('Left',6.99))

    def test_repeated_old_challenger_cannot_block_new_eligible_candidate(self):
        s=HandSelector();s.update('head',1,[observation()]);s.choose('Left',1.3)
        s.update('head',1.1,[])
        s.update('chest',1.,[observation(score=.99)]);s.choose('Left',1.4)
        s.update('right_hand',1.2,[observation(score=.95)])
        # Lost long enough for reacquisition, but old high-score candidate
        # cannot win over a new observation solely by reusing output time.
        s.missing_since['Left']=.5
        c=s.choose('Left',1.5)
        self.assertIsNotNone(c);self.assertEqual(c.stamp,1.2)


class AppearanceTimingTests(unittest.TestCase):
    def make_rigid(self,role):
        rec=TimelineRecording();b=BodyPreview(rec,[role],settings=load_settings())
        b.consume_imu(pose_packet(role,start=1,n=1),1.01)
        return b,rec

    def test_rigid_color_recovers_after_delayed_sample_behind_gray_event(self):
        for role in ('head','chest'):
            with self.subTest(role=role):
                b,rec=self.make_rigid(role);b.tick(1.6)
                root=f'/body/{role}/model'
                self.assertEqual(rec.latest(root+'/points','colors'),colors([110]*3))
                b.consume_imu(pose_packet(role,start=1.5,n=1),1.8)
                b.tick(1.8)
                # This late sample cannot override a later gray event.
                self.assertEqual(rec.latest(root+'/points','colors'),colors([110]*3))
                self.assertTrue(b.last_render[role]['grey'])
                b.consume_imu(pose_packet(role,start=1.7,n=1),1.9);b.tick(1.9)
                self.assertEqual(rec.latest(root+'/points','colors'),colors([105,215,235]))
                self.assertEqual(rec.latest(root+'/front','colors'),colors([255,190,70]))
                self.assertFalse(b.last_render[role]['grey'])
                self.assertIn('IMU: '+role,b.last_state[role])
                self.assertEqual(rec.latest('/imu/'+role+'/orientation/body_status','text'),[b.last_state[role]])

    def test_rigid_geometry_recovers_after_clear_with_delayed_first_packet(self):
        for role in ('head','chest'):
            b,rec=self.make_rigid(role);b.tick(3.)
            root=f'/body/{role}/model'
            b.consume_imu(pose_packet(role,start=2.9,n=1),3.1)
            self.assertIsNone(rec.latest(root+'/points','positions'))
            b.consume_imu(pose_packet(role,start=3.1,n=1),3.3)
            self.assertIsNotNone(rec.latest(root+'/points','positions'))
            self.assertIsNotNone(rec.latest(root,'mat3x3'))
            self.assertFalse(b.last_render[role]['hidden'])

    def test_gray_updates_only_color_never_overwrites_newer_hand_geometry(self):
        rec=TimelineRecording();b=BodyPreview(rec,['left_hand'],settings=load_settings())
        for stamp,bent in ((1.,False),(1.6,True)):
            p=pose_packet(start=stamp-.05,n=31)
            for row in p['imu']:row[2:]=increment([0,0,row[0]])
            b.consume_imu(p,stamp+.2)
            if bent:b.tick(1.7)
            b.consume_visual(dict(role='left_hand',received=stamp,image_device_time=stamp,
                                  observations=[observation(bent=bent)]),stamp+.2)
            b.tick(stamp+.2)
        root='/body/left_hand/model'
        # Gray at 1.7 must not write the OLD joint positions at 1.7, which
        # would win over bent joints and their paired quaternion at 1.6.
        self.assertAlmostEqual(rec.latest_time(root+'/points','positions'),1.6)
        self.assertAlmostEqual(rec.latest_time(root,'mat3x3'),1.6)
        np.testing.assert_allclose(rec.latest(root+'/points','positions'),b.last_render['left_hand']['shape'],atol=1e-7)
        self.assertEqual(rec.latest(root+'/points','colors'),colors([110]*3))

    def test_fresh_accepted_delayed_hand_not_immediately_grayed(self):
        rec=TimelineRecording();b=BodyPreview(rec,['left_hand'],settings=load_settings())
        b.consume_imu(pose_packet(start=1,n=151),1.76)
        b.consume_visual(dict(role='left_hand',received=1.2,image_device_time=1.2,
                              observations=[observation()]),1.76)
        b.tick(1.76)
        self.assertEqual(b.last_render['left_hand']['stamp'],1.2)
        self.assertFalse(b.last_render['left_hand']['grey'])
        # Expired data still goes gray, even while the IMU remains fresh.
        b.consume_imu(pose_packet(start=1.8,n=21),1.91);b.tick(1.91)
        self.assertTrue(b.last_render['left_hand']['grey'])
        self.assertEqual(b.last_render['left_hand']['stamp'],1.2)

    def test_summary_retains_per_role_wait_and_presentation_after_preview_exit(self):
        b,rec=self.make_rigid('chest');b.tick(1.6);summary=b.summary()
        self.assertEqual(summary['states']['chest'],'IMU STALE')
        self.assertTrue(summary['models']['chest']['grey'])
        self.assertEqual(summary['models']['chest']['stamp'],1.)
        self.assertEqual(summary['models']['chest']['appearance_stamp'],1.6)

if __name__=='__main__':unittest.main()
