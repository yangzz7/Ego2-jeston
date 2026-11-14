"""Geometry, identity, timing and loss handling of the derived dashboard."""
import copy
import math
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np
import rerun as rr
from ego2.common.mahony import rotation,increment
from ego2.multi_vio.hand_shape import template,palm_local,retarget,candidate,HandSelector,EDGES,PALM,pack_observations
from ego2.multi_vio.body_preview import PoseHistory,BodyPreview,DEFAULTS,load_settings,part_geometry,front_marker
from ego2.multi_vio.orientation_service import RoleProcessor


def observation(side='Left',eye='cam0',track=1,score=.95,bent=False):
    p=template(side)
    if bent:
        for root in (5,9,13,17):
            for j in range(1,4):
                length=np.linalg.norm(p[root+j]-p[root+j-1])
                angle=.5*j
                p[root+j]=p[root+j-1]+[0,length*math.cos(angle),length*math.sin(angle)]
    xy=np.column_stack((.5+p[:,0]*3,.35+p[:,1]*2,np.zeros(21)))
    return dict(side=side,eye=eye,track_id=track,handedness_score=score,
                points=xy,world_points=p.copy(),visible=np.ones(21,bool))


def pose_packet(role='left_hand',start=1.,n=21,offset=0.,valid=True):
    t=start+np.arange(n)*.005
    q=np.tile([1,0,0,0],(n,1)).astype(float)
    if not valid:q[:]=np.nan
    return dict(role=role,imu=np.column_stack((t,t,q)),clock=dict(clock_offset=offset,clock_epoch=0),
                body_reference=dict(segment=1,start=start,world_from_initial_camera_wxyz=increment([-math.pi/2,0,0]).tolist()) if valid else None,
                status=[(t[-1],'TRACKING' if valid else 'WAITING: reset')])


class GeometryTests(unittest.TestCase):
    def test_palm_local_removes_camera_rotation_translation_and_scale(self):
        for side in ('Left','Right'):
            p=observation(side,bent=True)['world_points']
            r=rotation(increment([.7,-.5,.8]))
            np.testing.assert_allclose(palm_local(p,side),palm_local(1.2*p@r.T+[.1,.02,-.03],side),atol=1e-12)
    def test_articulation_changes_and_bone_lengths_remain_fixed(self):
        for side in ('Left','Right'):
            a=retarget(palm_local(observation(side)['world_points'],side),side)
            c=retarget(palm_local(observation(side,bent=True)['world_points'],side),side)
            self.assertGreater(np.linalg.norm(a-c),.01)
            for i,j in EDGES: self.assertAlmostEqual(np.linalg.norm(a[j]-a[i]),np.linalg.norm(c[j]-c[i]),places=10)
            np.testing.assert_allclose(a[[0,5,9,13,17]].mean(axis=0),0,atol=1e-12)
            np.testing.assert_allclose(c[[0,5,9,13,17]].mean(axis=0),0,atol=1e-12)
    def test_bad_model_shape_or_crop_is_not_rendered(self):
        for change in ('nan','crop','basis','side'):
            item=observation()
            if change=='nan':item['world_points'][8]=np.nan
            if change=='crop':item['visible'][5]=False
            if change=='basis':item['world_points'][17]=item['world_points'][5]
            if change=='side':item['side']='Unknown'
            with self.assertRaises(ValueError):candidate('chest',1,item)
    def test_same_track_numbers_on_different_cameras_are_distinct(self):
        s=HandSelector();s.update('chest',1,[observation()]);s.update('head',1,[observation()])
        self.assertEqual(len(s.candidates),2)
    def test_source_handover_requires_sustained_quality_advantage(self):
        s=HandSelector(switch_margin=.01,switch_hold=.25)
        s.update('chest',1,[observation(score=.76)])
        self.assertEqual(s.choose('Left',1.01).source[0],'chest')
        for t in (1.05,1.15,1.25):
            s.update('chest',t,[observation(score=.76)])
            s.update('head',t,[observation(score=.99)])
            self.assertEqual(s.choose('Left',t+.01).source[0],'chest')
        s.update('chest',1.35,[observation(score=.76)])
        s.update('head',1.35,[observation(score=.99)])
        self.assertEqual(s.choose('Left',1.36).source[0],'head')
    def test_camera_role_does_not_determine_observed_hand(self):
        s=HandSelector();s.update('left_hand',1,[observation('Right')])
        self.assertIsNone(s.choose('Left',1.05));self.assertEqual(s.choose('Right',1.05).source[0],'left_hand')
    def test_missing_hand_switches_source_without_mixing_or_regressing_side(self):
        s=HandSelector();s.update('chest',1,[observation()]);self.assertEqual(s.choose('Left',1.01).source[0],'chest')
        s.update('head',1.02,[observation(score=.99)]);self.assertEqual(s.choose('Left',1.03).source[0],'chest')
        s.update('chest',1.04,[]);self.assertIsNone(s.choose('Left',1.05))
        for t in (1.12,1.22):
            s.update('head',t,[observation(score=.99)]);self.assertIsNone(s.choose('Left',t+.01))
        s.update('head',1.32,[observation(score=.99)])
        self.assertEqual(s.choose('Left',1.33).source[0],'head')
        self.assertIsNone(s.choose('Left',2))
    def test_ambiguous_same_side_tracks_in_one_eye_rejected(self):
        s=HandSelector();s.update('head',1,[observation(track=1),observation(track=2)])
        self.assertIsNone(s.choose('Left',1.02))
    def test_snapshot_does_not_change_original_raw_model_fields(self):
        from types import SimpleNamespace
        item=observation();item.update(handedness='Left',track_id=7)
        original=copy.deepcopy(item)
        hands=[[SimpleNamespace(id=7,handedness='Left')]]
        packed=pack_observations([[item]],hands,['cam0']);packed[0]['world_points'][:]=0
        np.testing.assert_array_equal(item['world_points'],original['world_points'])


class MountTests(unittest.TestCase):
    @staticmethod
    def logged_transform(rec,path):
        calls=[c for c in rec.log.call_args_list if c.args[0]==path and isinstance(c.args[1],rr.Transform3D)]
        t=calls[-1].args[1]
        # Rerun serializes Mat3x3 column-major.
        return np.asarray(t.mat3x3.as_arrow_array().to_pylist()[0]).reshape((3,3),order='F')

    def test_rendered_straight_hand_is_parallel_to_camera_forward(self):
        settings=load_settings()
        for role,side in (('left_hand','Left'),('right_hand','Right')):
            rec=MagicMock();preview=BodyPreview(rec,[role],settings=copy.deepcopy(settings))
            shape=retarget(palm_local(template(side),side),side)
            directions=[np.zeros(3),np.array([.3,-.6,.8])]
            directions += [axis*angle for axis in np.eye(3) for angle in (-math.pi/2,math.pi/2,math.pi)]
            for stamp,rv in enumerate(directions,1):
                q=increment(rv)
                preview._draw(role,float(stamp),q,shape,'test')
                model=self.logged_transform(rec,f'/body/{role}/model')
                camera=self.logged_transform(rec,f'/body/{role}/camera')
                forward=camera@np.array([0,0,1.])
                palm=camera@np.array([0,1.,0])
                # Check actual serialized points/transform, not just the
                # mounting config. Both hands and every camera orientation.
                points=[c.args[1] for c in rec.log.call_args_list if c.args[0]==f'/body/{role}/model/points'][-1]
                actual=np.asarray(points.positions.as_arrow_array().to_pylist())@model.T
                finger=actual[12]-actual[0];finger/=np.linalg.norm(finger)
                np.testing.assert_allclose(finger,forward,atol=1e-6)
                normal=model@np.array([0,0,-1.])
                np.testing.assert_allclose(normal,palm,atol=1e-6)
                self.assertAlmostEqual(float(finger@normal),0.,places=6)
                np.testing.assert_allclose(actual[PALM].mean(axis=0),0,atol=1e-8)
                np.testing.assert_allclose(model.T@model,np.eye(3),atol=1e-6)

    def test_rendered_hand_rolls_about_finger_when_camera_rolls_about_z(self):
        settings=load_settings()
        for role,side in (('left_hand','Left'),('right_hand','Right')):
            rec=MagicMock();preview=BodyPreview(rec,[role],settings=copy.deepcopy(settings))
            shape=retarget(palm_local(template(side),side),side)
            preview._draw(role,1.,increment([0,0,math.pi/2]),shape,'test')
            r=self.logged_transform(rec,f'/body/{role}/model')
            np.testing.assert_allclose(r@np.array([0,1.,0]),[0,0,1],atol=1e-6)
            np.testing.assert_allclose(r@np.array([0,0,-1.]),[-1,0,0],atol=1e-6)

    def test_hand_axes_follow_loss(self):
        settings=load_settings()
        rec=MagicMock();preview=BodyPreview(rec,['head','left_hand'],settings=settings)
        preview.consume_imu(pose_packet('head'),1.11);preview.consume_imu(pose_packet(),1.11)
        preview.consume_visual(dict(role='head',received=1.05,image_device_time=1.05,observations=[observation()]),1.15)
        preview.tick(1.15)
        axes=[c for c in rec.log.call_args_list if c.args[0]=='/body/left_hand/camera/axes']
        self.assertEqual(axes[-1].args[1].labels.as_arrow_array().to_pylist(),['X+','Y+ palm','Z+ forward'])
        self.assertFalse(any(c.kwargs.get('static',False) for c in axes))
        before=sum(c.args[0]=='/body/left_hand/camera' for c in rec.log.call_args_list)
        preview.tick(1.7)
        self.assertTrue(preview.last_render['left_hand']['grey'])
        # Staleness changes color, never moves axes to a newer IMU pose.
        self.assertEqual(sum(c.args[0]=='/body/left_hand/camera' for c in rec.log.call_args_list),before)
        preview.tick(3.)
        self.assertIsInstance([c.args[1] for c in rec.log.call_args_list if c.args[0]=='/body/left_hand/camera'][-1],rr.Clear)

    def test_head_chest_rendered_front_and_up_match_camera_axes(self):
        for role in ('head','chest'):
            rec=MagicMock();preview=BodyPreview(rec,[role],settings=load_settings())
            initial=self.logged_transform(rec,f'/body/{role}/model')
            np.testing.assert_allclose(initial@[0,1,0],[0,-1,0],atol=1e-6)
            directions=[np.zeros(3),np.array([.35,-.61,.72])]
            directions += [axis*angle for axis in np.eye(3) for angle in (-math.pi/2,math.pi/2,math.pi)]
            for stamp,rv in enumerate(directions,1):
                q=increment(rv);packet=pose_packet(role,start=float(stamp),n=1)
                packet['imu'][0,2:]=q
                preview.consume_imu(packet,stamp+.01)
                r=self.logged_transform(rec,f'/body/{role}/model')
                camera=self.logged_transform(rec,f'/body/{role}/camera')
                np.testing.assert_allclose(camera,rotation(q),atol=1e-6)
                np.testing.assert_allclose(r@[0,0,1],camera@[0,0,1],atol=1e-6)
                np.testing.assert_allclose(r@[0,1,0],camera@[0,-1,0],atol=1e-6)
                # Verify actual serialized geometry: head top vs neck, or
                # upper vs lower spine. Both must point opposite camera Y.
                points=[c.args[1] for c in rec.log.call_args_list if c.args[0]==f'/body/{role}/model/points'][-1]
                local=np.asarray(points.positions.as_arrow_array().to_pylist())
                up=local[0]-local[18 if role=='head' else 4];up/=np.linalg.norm(up)
                np.testing.assert_allclose(r@up,camera@[0,-1,0],atol=1e-6)
                mark=[c.args[1] for c in rec.log.call_args_list if c.args[0]==f'/body/{role}/model/front'][-1]
                front=np.asarray(mark.positions.as_arrow_array().to_pylist()[0])
                self.assertGreater(front[2],0.)
                base=front.copy();base[2]=0.
                forward=r@(front-base);forward/=np.linalg.norm(forward)
                np.testing.assert_allclose(forward,camera@[0,0,1],atol=1e-6)
                self.assertAlmostEqual(float(np.linalg.det(r)),1.,places=6)
                transform=[c.args[1] for c in rec.log.call_args_list if c.args[0]==f'/body/{role}/model'][-1]
                np.testing.assert_array_equal(transform.translation.as_arrow_array().to_pylist(),[[0,0,0]])

    def test_head_chest_axes_and_front_markers_follow_tracking_loss(self):
        for role in ('head','chest'):
            rec=MagicMock();preview=BodyPreview(rec,[role],settings=load_settings())
            preview.consume_imu(pose_packet(role),1.11)
            axes=[c for c in rec.log.call_args_list if c.args[0]==f'/body/{role}/camera/axes']
            expected=['X+','Y+ down','Z+ face' if role=='head' else 'Z+ chest']
            self.assertEqual(axes[-1].args[1].labels.as_arrow_array().to_pylist(),expected)
            self.assertFalse(any(c.args[0]==f'/body/{role}/reference' for c in rec.log.call_args_list))
            self.assertFalse(any(c.kwargs.get('static',False) for c in axes))
            preview.tick(1.7)
            self.assertTrue(preview.last_render[role]['grey'])
            mark=[c.args[1] for c in rec.log.call_args_list if c.args[0]==f'/body/{role}/model/front'][-1]
            grey=rr.Points3D([[0,0,0]],colors=[110,110,110])
            self.assertEqual(mark.colors.as_arrow_array().to_pylist(),grey.colors.as_arrow_array().to_pylist())
            preview.tick(3.)
            self.assertIsInstance([c.args[1] for c in rec.log.call_args_list if c.args[0]==f'/body/{role}/camera'][-1],rr.Clear)
            preview.consume_imu(pose_packet(role,start=3.1),3.21)
            self.assertFalse(preview.last_render[role]['hidden'])
            self.assertFalse(preview.last_render[role]['grey'])
            self.assertIsInstance([c.args[1] for c in rec.log.call_args_list if c.args[0]==f'/body/{role}/camera'][-1],rr.Transform3D)
            marks=[c for c in rec.log.call_args_list if c.args[0]==f'/body/{role}/model/front']
            self.assertFalse(any(c.kwargs.get('static',False) for c in marks))

    def test_rigid_models_have_visible_front_landmarks(self):
        for role in ('head','chest'):
            p,lines=part_geometry(role);front,label=front_marker(role)
            self.assertEqual(label,'FACE' if role=='head' else 'CHEST')
            self.assertGreater(front[2],0.)
            self.assertTrue(any(np.linalg.norm(point-front)<1e-9 for line in lines for point in line))
            if role=='chest':
                self.assertTrue(np.all(lines[0][:,2]<0))
                self.assertTrue(np.all(lines[-1][:,2]>0))
                self.assertLess(np.max(np.abs(p)),.15)

    def test_both_hands_middle_finger_and_palm_match_camera_axes(self):
        settings=load_settings()
        for role,side in (('left_hand','Left'),('right_hand','Right')):
            c=rotation(settings['roles'][role]['camera_from_body_wxyz'])
            middle=template(side)[9]-template(side)[0]
            middle/=np.linalg.norm(middle)
            np.testing.assert_allclose(c@middle,[0,0,1],atol=1e-12)
            np.testing.assert_allclose(c@np.array([0,0,-1]),[0,1,0],atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(c),1.)
            # Camera-axis rotations must turn the skeleton's corresponding
            # finger/palm axes, without mirroring or translating either hand.
            for axis in np.eye(3):
                body_rotation=c.T@rotation(increment(axis*.4))@c
                body_axis=c.T@axis
                np.testing.assert_allclose(body_rotation@body_axis,body_axis,atol=1e-12)
            np.testing.assert_allclose(c.T@rotation([1,0,0,0])@c,np.eye(3),atol=1e-12)

class TimingTests(unittest.TestCase):
    def test_slerp_matches_timestamp_and_shortest_sign(self):
        h=PoseHistory();p=pose_packet(n=2);p['imu'][1,0]=1.1;p['imu'][1,2:]=-increment([0,0,math.pi/2]);h.append(p['imu'])
        np.testing.assert_allclose(rotation(h.at(1.05)),rotation(increment([0,0,math.pi/4])),atol=1e-10)
        self.assertIsNone(h.at(1.2))
    def test_no_interpolation_across_nan_large_gap_or_reference_reset(self):
        h=PoseHistory();h.append(pose_packet(n=1)['imu']);h.append(pose_packet(start=1.3,n=1)['imu'])
        self.assertIsNone(h.at(1.1))
        h.append(pose_packet(start=1.4,n=1,valid=False)['imu']);self.assertIsNone(h.at(1.3))
        h.append(pose_packet(start=1.5,n=1)['imu']);self.assertIsNone(h.at(1.3));self.assertIsNotNone(h.at(1.5))
    def test_combination_uses_visual_capture_time_and_fixed_translation(self):
        rec=MagicMock();b=BodyPreview(rec,['head','left_hand'],settings=copy.deepcopy(DEFAULTS))
        b.consume_imu(pose_packet(),1.11);b.consume_imu(pose_packet('head'),1.11)
        packet=dict(role='head',received=1.05,image_device_time=1.05,observations=[observation()])
        b.consume_visual(packet,1.15);b.tick(1.15)
        self.assertAlmostEqual(b.last_render['left_hand']['stamp'],1.05)
        calls=[c for c in rec.log.call_args_list if c.args[0]=='/body/left_hand/model']
        t=calls[-1].args[1]
        np.testing.assert_array_equal(t.translation.as_arrow_array().to_pylist(),[[0,0,0]])
        self.assertIn('head/cam0/1',b.last_state['left_hand'])
        b.tick(1.7);self.assertTrue(b.last_render['left_hand']['grey'])
        b.tick(3);self.assertTrue(b.last_render['left_hand']['hidden'])
    def test_bad_clock_no_pose_and_old_visual_fail_closed(self):
        for kind in ('clock','pose','old'):
            b=BodyPreview(MagicMock(),['head','left_hand'],settings=copy.deepcopy(DEFAULTS))
            b.consume_imu(pose_packet('head'),1.11)
            b.consume_imu(pose_packet(valid=kind!='pose'),1.11)
            p=dict(role='head',received=1.05,image_device_time=2 if kind=='clock' else 1.05,observations=[observation()])
            b.consume_visual(p,3 if kind=='old' else 1.15);b.tick(3 if kind=='old' else 1.15)
            self.assertNotIn('left_hand',b.last_render)
    def test_valid_imu_does_not_rotate_old_hand_shape(self):
        rec=MagicMock();b=BodyPreview(rec,['left_hand'],settings=copy.deepcopy(DEFAULTS))
        b.consume_imu(pose_packet(),1.11)
        b.consume_visual(dict(role='left_hand',received=1.05,image_device_time=1.05,
                             observations=[observation()]),1.15);b.tick(1.15)
        count=lambda:sum(c.args[0]=='/body/left_hand/model' for c in rec.log.call_args_list)
        before=count()
        p=pose_packet(start=1.6);p['imu'][:,2:]=increment([0,0,1.])
        b.consume_imu(p,1.71);b.tick(1.71)
        self.assertEqual(count(),before)
        self.assertTrue(b.last_render['left_hand']['grey'])
    def test_initial_body_geometry_does_not_override_tracking_colors(self):
        rec=MagicMock();b=BodyPreview(rec,['head'],settings=copy.deepcopy(DEFAULTS))
        b.consume_imu(pose_packet('head'),1.11)
        calls=[c for c in rec.log.call_args_list if '/model/' in c.args[0]]
        self.assertTrue(calls)
        self.assertFalse(any(c.kwargs.get('static',False) for c in calls))
        self.assertFalse(b.last_render['head']['grey'])
    def test_real_rerun_serialization_for_articulated_hand_and_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=rr.RecordingStream('body-test');path=Path(tmp)/'body.rrd';r.save(path)
            b=BodyPreview(r,['head','chest','left_hand','right_hand'],settings=load_settings())
            for role in ('head','chest','left_hand','right_hand'):b.consume_imu(pose_packet(role),1.11)
            b.consume_visual(dict(role='head',received=1.05,image_device_time=1.05,observations=[observation(bent=True)]),1.15)
            b.consume_visual(dict(role='chest',received=1.05,image_device_time=1.05,observations=[observation('Right',bent=True)]),1.15)
            b.tick(1.15);b.tick(1.8);b.tick(3);r.flush();r.disconnect()
            self.assertGreater(path.stat().st_size,1000)


class RawSidecarTests(unittest.TestCase):
    def test_raw_samples_exact_even_while_fusion_is_waiting_and_queue_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal=Path(tmp)/'cam.yaml';cal.write_text('cam0:\n  T_cam_imu: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n')
            q=queue.Queue(maxsize=1);q.put('busy');p=RoleProcessor(q,'head',0,{'calibration':{'file':str(cal)}})
            t=1+np.arange(12)*.005;values=np.column_stack((t,t,np.ones((12,6))))
            before=values.copy();packet=dict(imu=values,received=2.,clock_offset=0,clock_epoch=0,device_image=1.1)
            p.consume(packet,2.);np.testing.assert_array_equal(before,values);q.get();p.publisher.flush(2.2)
            out=q.get();np.testing.assert_array_equal(out['raw_imu'][:,2:],values[:,2:]);self.assertTrue(np.isnan(out['imu'][:,2:]).all())
            self.assertEqual(out['clock']['clock_offset'],0)
    def test_raw_gap_is_marked_without_removing_valid_measurements(self):
        from ego2.multi_vio.dashboard_transport import DashboardPublisher
        p=DashboardPublisher(queue.Queue(),'head',0,{'calibration':{'file':'/missing'}})
        rows=np.array([[1,1,1,2,3,4,5,6],[1.1,1.1,7,8,9,10,11,12.]])
        p.append_raw(rows,{})
        raw=np.asarray(p.raw.rows);self.assertEqual(len(raw),3);self.assertTrue(np.isnan(raw[1,2:]).all())
        np.testing.assert_array_equal(raw[[0,2]],rows)

if __name__=='__main__':unittest.main()
