"""Regression: dead source pinning and a camera initialized looking upward."""
import copy
import math
import os
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ego2.common.mahony import Mahony, increment, rotation, from_rotation, conjugate, multiply
from ego2.multi_vio.hand_shape import HandSelector
from ego2.multi_vio.body_preview import BodyPreview, gravity_display_frame, load_settings
from ego2.multi_vio.native_orientation import NativeQuaternionPublisher
from tests.test_body_preview import observation, pose_packet
from tests import test_body_preview as fixtures


class RecoveryTests(unittest.TestCase):
    def test_track_churn_and_rejected_frames_cannot_pin_dead_source(self):
        for side in ('Left','Right'):
            s=HandSelector();s.update('head',1,[observation(side)]);s.choose(side,1.01)
            s.update('head',1.1,[])
            received=[]
            for i in range(1,40):
                t=1.1+i*.1
                # Every second frame is rejected; every valid frame has a new
                # track ID, so sustained same-ID confirmation is impossible.
                s.update('chest',t,[observation(side,track=i,score=.5 if i%2 else .95)])
                c=s.choose(side,t+.01)
                if c:received.append(c.stamp)
            self.assertTrue(received)
            self.assertLessEqual(received[0],1.91)
            self.assertLessEqual(max(np.diff(received)),.81)
            self.assertGreater(s.reasons['source_reacquired'],0)
            self.assertEqual(s.accepted_sides[side],20)

    def test_deadline_cannot_make_stale_or_rejected_candidate_valid(self):
        s=HandSelector();s.update('head',1,[observation()]);s.choose('Left',1.01)
        s.update('head',1.1,[]);s.update('chest',1.6,[observation()])
        self.assertIsNone(s.choose('Left',2.3))  # capture age exceeds .65 s
        s.update('chest',2.35,[observation(score=.5)])
        self.assertIsNone(s.choose('Left',2.36))
        s.update('chest',2.4,[observation(track=9)])
        self.assertEqual(s.choose('Left',2.71).stamp,2.4)

    def test_recovery_after_hidden_model_keeps_matching_pose_and_no_translation(self):
        b=BodyPreview(MagicMock(),['head','left_hand'],settings=load_settings())
        for r in b.roles:b.consume_imu(pose_packet(r),1.11)
        b.consume_visual(dict(role='head',received=1.05,image_device_time=1.05,observations=[observation()]),1.15)
        b.tick(1.15)
        b.consume_visual(dict(role='head',received=1.2,image_device_time=1.2,observations=[]),1.21)
        b.tick(1.21);b.tick(3)
        self.assertTrue(b.last_render['left_hand']['hidden'])
        for r in b.roles:
            p=pose_packet(r,start=3.1)
            p['imu'][:,2:]=increment([.2,-.3,.4])
            b.consume_imu(p,3.21)
        b.consume_visual(dict(role='head',received=3.15,image_device_time=3.15,observations=[observation(track=12)]),3.22)
        b.tick(3.22)
        self.assertFalse(b.last_render['left_hand']['hidden'])
        self.assertEqual(b.last_render['left_hand']['stamp'],3.15)
        expected=rotation(increment([.2,-.3,.4]))@b.mounts['left_hand']
        np.testing.assert_allclose(fixtures.MountTests.logged_transform(b.recording,'/body/left_hand/model'),expected,atol=1e-6)


class GravityTests(unittest.TestCase):
    def test_upward_start_then_forward_up_and_down_match_physical_front(self):
        # World uses Z up; display uses Y down, Z forward. Camera pitched up
        # at initialization must not be silently drawn facing horizontally.
        world_from_display=rotation(increment([-math.pi/2,0,0]))
        for role in ('head','chest'):
            for start_pitch in (0,math.pi/2,-math.pi/2,.5):
                initial=rotation(increment([start_pitch,0,0]))
                ref=from_rotation(world_from_display@initial)
                b=BodyPreview(MagicMock(),[role],settings=load_settings())
                for stamp,pitch in enumerate((start_pitch,0,math.pi/2,-math.pi/2,.4),1):
                    target=rotation(increment([pitch,0,0]))
                    p=pose_packet(role,start=stamp,n=1)
                    p['body_reference']=dict(segment=1,start=1.,world_from_initial_camera_wxyz=ref.tolist())
                    p['imu'][0,2:]=from_rotation(initial.T@target)
                    before=p['imu'].copy();b.consume_imu(p,stamp+.01)
                    actual=fixtures.MountTests.logged_transform(b.recording,f'/body/{role}/model')
                    np.testing.assert_allclose(actual@[0,0,1],target@[0,0,1],atol=1e-6)
                    np.testing.assert_allclose(actual@[0,1,0],target@[0,-1,0],atol=1e-6)
                    np.testing.assert_array_equal(p['imu'],before)

    def test_gravity_frame_is_proper_at_vertical_and_roll_extremes(self):
        for axis in np.eye(3):
            for angle in (0,math.pi/2,-math.pi/2,math.pi,.43):
                q=increment(axis*angle);frame=gravity_display_frame(q)
                np.testing.assert_allclose(frame.T@frame,np.eye(3),atol=1e-10)
                self.assertAlmostEqual(np.linalg.det(frame),1.)
                np.testing.assert_allclose(frame@rotation(q).T@[0,0,1],[0,-1,0],atol=1e-10)

    def test_missing_or_future_reference_does_not_guess_initial_pose(self):
        for ref in (None,dict(segment=2,start=4.,world_from_initial_camera_wxyz=[1,0,0,0])):
            b=BodyPreview(MagicMock(),['chest'],settings=load_settings())
            p=pose_packet('chest');p['body_reference']=ref
            b.consume_imu(p,1.11);b.tick(1.11)
            self.assertNotIn('chest',b.last_render)
            self.assertEqual(b.last_state['chest'],'WAITING: gravity reference')

    def test_new_reference_after_gap_does_not_reuse_old_tilt(self):
        b=BodyPreview(MagicMock(),['chest'],settings=load_settings())
        b.consume_imu(pose_packet('chest'),1.11)
        b.consume_imu(pose_packet('chest',start=1.2,n=1,valid=False),1.21)
        p=pose_packet('chest',start=4.,n=1)
        p['body_reference']=dict(segment=2,start=4.,world_from_initial_camera_wxyz=[1,0,0,0])
        b.consume_imu(p,4.01)
        r=fixtures.MountTests.logged_transform(b.recording,'/body/chest/model')
        np.testing.assert_allclose(r@[0,0,1],[0,-1,0],atol=1e-6)


class ReferenceTransportTests(unittest.TestCase):
    def test_native_raw_imu_to_chest_model_with_nonidentity_camera_extrinsic(self):
        with tempfile.TemporaryDirectory() as d:
            r_ci=rotation(increment([.3,-.4,.6]));transform=np.eye(4);transform[:3,:3]=r_ci
            import json
            cal=Path(d)/'cam.yaml';cal.write_text('cam0:\n  T_cam_imu: '+json.dumps(transform.tolist())+'\n')
            with patch.dict(os.environ,EGO2_ATTITUDE_BACKEND='native'):
                p=NativeQuaternionPublisher(queue.Queue(),'chest',0,{'calibration':{'file':str(cal)}},capacity=4096)
            rows=[]
            for t in np.arange(1001)*.005:
                angle=-min(max(t-3.,0.),2.)*math.pi/4
                world_from_imu=rotation(increment([angle,0,0]))@r_ci
                gyro=r_ci.T@np.array([-math.pi/4 if 3.<t<=5. else 0,0,0])
                rows.append([t,t,*(world_from_imu.T@[0,0,9.80665]),*gyro])
            for batch in np.array_split(np.asarray(rows),51):p.append_rows(batch)
            packet=p._packet(5.);packet['clock']=dict(clock_offset=0,clock_epoch=0)
            b=BodyPreview(MagicMock(),['chest'],settings=load_settings());b.consume_imu(packet,5.01)
            r=fixtures.MountTests.logged_transform(b.recording,'/body/chest/model')
            np.testing.assert_allclose(r@[0,0,1],[0,0,1],atol=.01)
            np.testing.assert_allclose(r@[0,1,0],[0,-1,0],atol=.01)

    def test_python_and_native_keep_quaternions_exact_and_publish_segment_reference(self):
        with tempfile.TemporaryDirectory() as d:
            cal=Path(d)/'cam.yaml'
            cal.write_text('cam0:\n  T_cam_imu: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n')
            for backend in ('python','native'):
                with patch.dict(os.environ,EGO2_ATTITUDE_BACKEND=backend):
                    p=NativeQuaternionPublisher(queue.Queue(),'chest',0,{'calibration':{'file':str(cal)}},capacity=4096)
                expected=Mahony(np.eye(3))
                # Lens upward at startup: identity world reference. Include
                # tilted initialization after a long gap in the second block.
                for base,acc in ((0,[0,0,9.80665]),(6,[0,-9.80665,0])):
                    t=base+np.arange(620)*.005
                    rows=np.column_stack((t,t,np.tile(acc,(len(t),1)),np.zeros((len(t),3))))
                    original=rows.copy();qs=[]
                    for row in rows:
                        q=expected.update(row[1],row[2:5],row[5:]);qs.append(q if q is not None else np.full(4,np.nan))
                    for batch in np.array_split(rows,31):p.append_rows(batch)
                    actual=np.asarray(p.rows)[-len(t):,2:]
                    np.testing.assert_allclose(actual,qs,atol=1e-10,equal_nan=True)
                    np.testing.assert_array_equal(rows,original)
                    packet=p._packet(float(t[-1]));ref=packet['body_reference']
                    self.assertEqual(ref['segment'],expected.segment)
                    self.assertGreaterEqual(ref['start'],base+2.49)
                    self.assertLessEqual(ref['start'],base+2.7)
                    np.testing.assert_allclose(rotation(ref['world_from_initial_camera_wxyz']),rotation(expected.reference),atol=1e-10)
                p._append((10.,math.nan,*([math.nan]*6)))
                self.assertIsNone(p._packet(10.)['body_reference'])

if __name__=='__main__':unittest.main()
