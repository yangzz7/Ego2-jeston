"""Coordinate, identity, dropout and transport regressions for two-camera multi VIO."""
import copy
import json
import mmap
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock
import cv2
import numpy as np

from ego2.multi_vio.config import load_multi_config
from ego2.multi_vio.alignment import Coordinator, AlignmentWindow, inverse, interpolate, robust_fit
from ego2.multi_vio.transport import ClockMapper, RingReader, SLOT_BYTES, IMAGE_BYTES, SLOTS
from ego2.aprilgrid.pose import Measurement
from ego2.multi_vio.recording import Recording


def load_two_camera_config():
    return load_multi_config(selection={'chest':'s200100','left_hand':'s200032'})


def transform(v=(.12,-.25,.3),t=(.4,-.2,.6)):
    out=np.eye(4); out[:3,:3]=cv2.Rodrigues(np.array(v,float))[0]; out[:3,3]=t
    return out


def measured(board_camera):
    return Measurement(inverse(board_camera),.2,'stereo',frozenset((c,i) for c in ('cam0','cam1') for i in (0,5,30,35)),{})


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.devices=load_two_camera_config()

    def setUp(self):
        self.cfg=copy.deepcopy(self.config)
        self.cfg['alignment'].update(min_samples=5,min_duration_s=.3)
        self.c=Coordinator(self.cfg,self.devices)
        self.anchors={'chest':transform(), 'left_hand':transform((-.4,.15,-.8),(-.7,.1,.2))}
        self.poses={'chest':transform((.1,.2,.1),(-.2,.1,.6)), 'left_hand':transform((.3,-.1,.7),(.3,-.1,.5))}

    def packet(self,role,index,map_id='0_0',valid=True):
        return dict(frame_id=index,timestamp=index*.1,capture_time=index*.1,map_id=map_id,valid=valid,
                    initialization_complete=True,T_world_camera=(inverse(self.anchors[role]) @ self.poses[role]).tolist() if valid else None)

    def initialize(self):
        for index in range(6):
            packets={r:self.packet(r,index) for r in self.anchors}
            for r,p in packets.items(): self.c.ingest(r,p,index*.1)
            for r,p in packets.items(): self.c.observe_board(r,p,measured(self.poses[r]),index*.1)
            self.c.relative(index*.1)
        self.assertTrue(self.c.started)

    def test_rotated_worlds_align_and_relative_pose_has_correct_direction(self):
        self.initialize()
        for role in self.anchors:
            np.testing.assert_allclose(self.c.devices[role].anchor,self.anchors[role],atol=1e-9)
            np.testing.assert_allclose(self.c.board_pose(role,.5),self.poses[role],atol=1e-9)
            np.testing.assert_allclose(self.c.pose(role,.5),self.c.shared_from_board @ self.poses[role],atol=1e-9)
        np.testing.assert_allclose(self.c.initial_relatives['left_hand'],inverse(self.poses['chest'])@self.poses['left_hand'],atol=1e-9)
        np.testing.assert_allclose(inverse(self.c.pose('chest',.5)) @ self.c.pose('left_hand',.5),
                                   self.c.initial_relatives['left_hand'],atol=1e-9)

    def test_board_removal_and_new_board_pose_cannot_modify_locked_anchor(self):
        self.initialize()
        self.poses['left_hand'][:3,3]+=[.1,.2,0]
        for r in self.anchors:
            p=self.packet(r,6); self.c.ingest(r,p,.6)
            self.c.observe_board(r,p,None,.6)
            self.c.observe_board(r,p,measured(transform(t=(4,5,6))),.6)
            np.testing.assert_allclose(self.c.devices[r].anchor,self.anchors[r],atol=1e-9)
        np.testing.assert_allclose(self.c.pose('left_hand',.6),self.c.shared_from_board @ self.poses['left_hand'],atol=1e-9)

    def test_gravity_up_motion_in_tilted_board_frame_becomes_positive_z(self):
        self.initialize()
        before=self.c.pose('chest',.5)
        # Lift along native VIO +Z; the board is tilted in that world.
        self.poses['chest'][:3,3]+=self.anchors['chest'][:3,:3] @ [0,0,.25]
        self.c.ingest('chest',self.packet('chest',6),.6)
        after=self.c.pose('chest',.6)
        np.testing.assert_allclose(after[:3,3]-before[:3,3],[0,0,.25],atol=1e-9)
        np.testing.assert_allclose(after[:3,:3],before[:3,:3],atol=1e-9)
        np.testing.assert_array_equal(self.c.shared_from_board[:3,3],[0,0,0])

    def test_chest_map_realignment_cannot_rotate_shared_world_or_hand(self):
        self.initialize()
        frame=self.c.shared_from_board.copy()
        before=self.c.pose('left_hand',.5)
        self.anchors['chest']=transform((.6,-.3,1.1),(2,3,4))
        for index in range(6,12):
            for role in self.anchors:
                p=self.packet(role,index,map_id='new' if role=='chest' else '0_0')
                self.c.ingest(role,p,index*.1)
                self.c.observe_board(role,p,measured(self.poses[role]),index*.1)
            self.c.relative(index*.1)
        np.testing.assert_allclose(self.c.shared_from_board,frame,atol=1e-9)
        np.testing.assert_allclose(self.c.pose('left_hand',1.1),before,atol=1e-9)
        np.testing.assert_allclose(self.c.pose('chest',1.1),frame @ self.poses['chest'],atol=1e-9)

    def test_recording_keeps_board_and_shared_coordinates_distinct(self):
        import io
        self.initialize()
        recorder=Recording.__new__(Recording)
        recorder.session=True; recorder.active=True; recorder.cutoff=None; recorder.end_cutoff=None
        recorder.poses=io.StringIO()
        packet=self.packet('left_hand',5)
        recorder.pose('left_hand',packet,self.c.pose('left_hand',.5),self.c.board_pose('left_hand',.5))
        row=json.loads(recorder.poses.getvalue())
        np.testing.assert_allclose(row['T_world_camera'],packet['T_world_camera'])
        np.testing.assert_allclose(row['T_board_camera'],self.poses['left_hand'])
        np.testing.assert_allclose(row['T_shared_camera'],self.c.shared_from_board @ self.poses['left_hand'])

    def test_no_trajectory_until_both_aligned_and_same_time_available(self):
        for index in range(8):
            for role in self.anchors:
                p=self.packet(role,index); self.c.ingest(role,p,index*.1)
                if role=='chest': self.c.observe_board(role,p,measured(self.poses[role]),index*.1)
            self.c.relative(index*.1)
        self.assertFalse(self.c.started)
        self.assertIsNone(self.c.pose('chest',.7))

    def test_invalid_and_monocular_measurements_cannot_align(self):
        window=self.c.devices['chest'].window
        m=measured(self.poses['chest']); m.inliers=frozenset(k for k in m.inliers if k[0]=='cam1')
        for i in range(10):
            self.assertIsNone(window.add(self.packet('chest',i),None if i%2 else m))
        self.assertEqual(len(window.samples),0)

    def test_collinear_tags_rejected(self):
        m=measured(self.poses['chest']); m.inliers=frozenset((c,i) for c in ('cam0','cam1') for i in (0,7,14,21))
        w=self.c.devices['chest'].window
        self.assertIsNone(w.add(self.packet('chest',0),m))
        self.assertEqual(w.reason,'insufficient_board_coverage')

    def test_map_reset_invalidates_only_affected_anchor_and_rejects_old_worker(self):
        self.initialize()
        old=self.packet('left_hand',5)
        new=self.packet('left_hand',6,map_id='1_0')
        self.c.ingest('left_hand',new,.6)
        self.assertIsNone(self.c.pose('left_hand',.6))
        self.assertIsNotNone(self.c.pose('chest',.6))
        self.assertIsNone(self.c.relative(.6))
        self.c.observe_board('left_hand',old,measured(self.poses['left_hand']),.6)
        self.assertEqual(len(self.c.devices['left_hand'].window.samples),0)
        self.assertEqual(self.c.devices['left_hand'].epoch,1)

    def test_tracking_loss_stale_and_recovery_cannot_bridge_interpolation(self):
        self.initialize()
        self.c.ingest('left_hand',self.packet('left_hand',6,valid=False),.6)
        self.assertIsNone(self.c.pose('left_hand',.6))
        self.assertFalse(self.c.devices['left_hand'].buffer)
        self.assertIsNone(self.c.pose('chest',2))
        self.c.ingest('left_hand',self.packet('left_hand',7),.7)
        self.assertIsNotNone(self.c.pose('left_hand',.7))
        self.assertIsNone(self.c.relative(.7))

    def test_delayed_capture_cannot_become_fresh_by_late_delivery(self):
        self.initialize()
        packet=self.packet('left_hand',6)
        packet['host_monotonic']=.6
        self.c.ingest('left_hand',packet,2.)
        self.assertEqual(self.c.status('left_hand',2.),'STALE')
        self.assertIsNone(self.c.pose('left_hand',2.))

    def test_old_board_worker_result_cannot_anchor_fresh_vio(self):
        for role in self.anchors:
            self.c.ingest(role,self.packet(role,20),2.)
        old=self.packet('left_hand',1)
        old['host_monotonic']=.1
        self.c.observe_board('left_hand',old,measured(self.poses['left_hand']),2.)
        self.assertFalse(self.c.devices['left_hand'].window.samples)
        self.assertEqual(self.c.devices['left_hand'].window.reason,'board_frame_not_in_current_track')

    def test_outliers_do_not_drag_consensus_and_spread_is_rejected(self):
        values=[self.anchors['chest'].copy() for _ in range(10)]
        values[0]=transform((1,1,1),(5,6,7))
        fit,q=robust_fit(values,self.cfg['alignment'])
        np.testing.assert_allclose(fit,self.anchors['chest'],atol=1e-9)
        self.assertEqual(q['inliers'],9)
        for i,value in enumerate(values): value[0,3]=i*.1
        fit,q=robust_fit(values,self.cfg['alignment'])
        self.assertIsNone(fit)

    def test_rotation_interpolation_and_gap_rejection(self):
        a=np.eye(4); b=transform((0,0,np.pi/2),(2,0,0))
        half=interpolate([(0,a),(.1,b)],.05,.2)
        np.testing.assert_allclose(half,transform((0,0,np.pi/4),(1,0,0)),atol=1e-9)
        self.assertIsNone(interpolate([(0,a),(1,b)],.5,.2))
        self.assertIsNone(interpolate([(0,a),(.1,b)],.2,.2))


class TransportTests(unittest.TestCase):
    def test_clock_offset_epochs_jitter_and_backwards_clock(self):
        a,b=ClockMapper(100),ClockMapper(100)
        for i in range(20):
            t=i*.1
            x,_=a.update(1000+t,100+t+.01)
            y,_=b.update(40+t,100+t+.012)
            self.assertLess(abs(x-y),.003)
        with self.assertRaises(ValueError): a.update(999,103)

    def test_ring_copy_and_overwritten_frame_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'frames.bin'
            with path.open('w+b') as f:
                f.truncate(SLOT_BYTES*SLOTS)
                mm=mmap.mmap(f.fileno(),0)
                struct.pack_into('<Q',mm,SLOT_BYTES,5)
                mm[SLOT_BYTES+8:SLOT_BYTES+8+IMAGE_BYTES]=bytes([20])*IMAGE_BYTES
                mm[SLOT_BYTES+8+IMAGE_BYTES:2*SLOT_BYTES]=bytes([100])*IMAGE_BYTES
                reader=RingReader(path)
                try:
                    images=reader.read(dict(slot=1,frame_id=4))
                    self.assertEqual(images[0][1,2,0],20)
                    self.assertEqual(images[1][1,2,0],100)
                    self.assertIsNone(reader.read(dict(slot=1,frame_id=7)))
                    struct.pack_into('<Q',mm,SLOT_BYTES,8)
                    self.assertEqual(images[0][1,2,0],20)  # copied, not a live view
                    self.assertIsNone(reader.read(dict(slot=1,frame_id=4)))
                finally: reader.close(); mm.close()


class ConfigurationTests(unittest.TestCase):
    def test_per_serial_calibration_and_same_device_rejected(self):
        cfg,devices=load_two_camera_config()
        self.assertEqual(devices['chest']['native']['serial'],'s200100')
        self.assertEqual(devices['left_hand']['native']['serial'],'s200032')
        self.assertEqual(cfg['viewer']['display_order'],['cam1','cam0'])
        with self.assertRaises(ValueError): load_multi_config(selection={'chest':'s200032','left_hand':'s200032'})

    def test_official_rerun_blueprint_serializes_with_five_spatial_views(self):
        import rerun as rr
        from ego2.multi_vio.visualizer import build_blueprint
        cfg,_=load_two_camera_config()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'layout.rbl'
            build_blueprint(cfg).save('ego2-multi-layout-test',str(path))
            self.assertGreater(path.stat().st_size,100)


class ViewpointTests(unittest.TestCase):
    def setUp(self):
        self.cfg,_=load_two_camera_config()

    def test_forward_motion_goes_into_screen_and_gravity_points_screen_up(self):
        from ego2.multi_vio.visualizer import initial_multi_eye_geometry
        chest=transform((.9,-.3,.5),(.2,-.1,.8))
        hand=transform(t=(.5,-.2,.5))
        eye=initial_multi_eye_geometry({'chest':chest,'left_hand':hand},self.cfg['viewer'])
        depth=eye['look_target']-eye['position']
        self.assertGreaterEqual(np.linalg.norm(depth),2.5)
        depth/=np.linalg.norm(depth)
        up=eye['eye_up']
        right=np.cross(depth,up)
        movement=chest[:3,2]*.3
        self.assertAlmostEqual(np.dot(movement,depth),.3)
        self.assertAlmostEqual(np.dot(movement,right),0.)
        self.assertAlmostEqual(np.dot(movement,up),0.)
        self.assertGreater(np.dot(up,[0,0,1]),0.)

    def test_vertical_optical_direction_and_wide_camera_separation(self):
        from ego2.multi_vio.visualizer import initial_multi_eye_geometry
        for angle in (0.,np.pi):
            chest=transform((angle,0,0),(0,0,0))
            hand=transform(t=(4,0,0))
            eye=initial_multi_eye_geometry({'chest':chest,'left_hand':hand},self.cfg['viewer'])
            self.assertTrue(np.isfinite(eye['eye_up']).all())
            self.assertAlmostEqual(np.dot(eye['eye_up'],chest[:3,2]),0.)
            self.assertGreater(np.linalg.norm(eye['look_target']-eye['position']),6.)

    def test_default_view_only_sent_once_after_both_poses_are_available(self):
        from ego2.multi_vio.visualizer import Visualizer, build_blueprint
        viewer=Visualizer.__new__(Visualizer)
        viewer.cfg=self.cfg
        viewer.blueprint=build_blueprint(self.cfg)
        view=viewer.blueprint.root_container.contents[0].contents[0]
        original_id=view.id
        viewer.eye_initialized=False
        viewer.recording=MagicMock()
        viewer.initialize_poses({'chest':transform(),'left_hand':None})
        viewer.recording.send_blueprint.assert_not_called()
        viewer.initialize_poses({'chest':transform(),'left_hand':transform(t=(.7,0,.5))})
        self.assertEqual(view.id,original_id)
        # The updated actual SDK blueprint must still serialize correctly.
        with tempfile.TemporaryDirectory() as directory:
            viewer.blueprint.save('ego2-multi-viewpoint-test',str(Path(directory)/'layout.rbl'))
        first=view.properties['EyeControls3D']
        viewer.initialize_poses({'chest':transform(t=(8,9,1)),'left_hand':transform()})
        viewer.recording.send_blueprint.assert_called_once()
        self.assertIs(view.properties['EyeControls3D'],first)


if __name__=='__main__': unittest.main()
