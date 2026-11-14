"""Four-camera mapping, common-time alignment, frozen deltas and isolation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import yaml

from ego2.multi_vio.config import load_multi_config, MULTI_CONFIG, MULTI_ROLES
from ego2.multi_vio.alignment import Coordinator, inverse, relative_delta
from ego2.multi_vio.recording import Recording
from ego2.multi_vio.transport import check_devices
from ego2.multi_vio.visualizer import build_blueprint, initial_multi_eye_geometry, relative_document, Visualizer
from tests.test_multi_vio_core import transform, measured


class FourCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.devices = load_multi_config(demo=True)

    def setUp(self):
        self.cfg = copy.deepcopy(self.config)
        self.cfg['alignment'].update(min_samples=5, min_duration_s=.3)
        self.c = Coordinator(self.cfg, self.devices)
        self.anchors = {r: transform((.12+i*.17, -.25, .3-i*.28), (i*.7, -.2, .6))
                        for i,r in enumerate(MULTI_ROLES)}
        self.poses = {r: transform((.1+i*.1, -.2+i*.15, .25-i*.2), (i*.2, .3-i*.12, .6))
                      for i,r in enumerate(MULTI_ROLES)}

    def packet(self, role, index, *, stamp=None, valid=True, map_id='0'):
        stamp = index*.1 if stamp is None else stamp
        return dict(frame_id=index, timestamp=stamp, capture_time=stamp, map_id=map_id,
                    initialization_complete=True, valid=valid,
                    T_world_camera=(inverse(self.anchors[role]) @ self.poses[role]).tolist() if valid else None)

    def step(self, index, skip=None):
        packets={r:self.packet(r,index) for r in MULTI_ROLES}
        for r,p in packets.items(): self.c.ingest(r,p,index*.1)
        for r,p in packets.items():
            if r != skip: self.c.observe_board(r,p,measured(self.poses[r]),index*.1)
        self.c.relative(index*.1)

    def initialize(self):
        for i in range(6): self.step(i)
        self.assertTrue(self.c.started)

    def test_four_independent_worlds_and_three_correct_relative_transforms(self):
        self.initialize()
        self.assertEqual(set(self.c.initial_relatives), {'head','left_hand','right_hand'})
        for r in MULTI_ROLES:
            np.testing.assert_allclose(self.c.devices[r].anchor,self.anchors[r],atol=1e-9)
            expected=self.c.shared_from_board @ self.poses[r]
            np.testing.assert_allclose(self.c.pose(r,.5),expected,atol=1e-9)
            if r != 'chest':
                np.testing.assert_allclose(self.c.initial_relatives[r],inverse(self.poses['chest']) @ self.poses[r],atol=1e-9)
                d=self.c.initial_deltas[r]
                ref=self.c.initial_poses['chest']
                np.testing.assert_allclose(d['delta_p_m'],expected[:3,3]-ref[:3,3],atol=1e-9)
                # Check relative rotation against an independently reconstructed
                # Euler rotation; this catches order and frame-convention errors.
                roll,pitch,yaw=np.deg2rad(d['delta_rpy_deg'])
                R=cv2.Rodrigues(np.array([0.,0.,yaw]))[0] @ cv2.Rodrigues(np.array([0.,pitch,0.]))[0] @ cv2.Rodrigues(np.array([roll,0.,0.]))[0]
                np.testing.assert_allclose(R @ ref[:3,:3],expected[:3,:3],atol=1e-9)

    def test_fourth_camera_is_required_before_any_trajectory(self):
        for i in range(8): self.step(i,skip='right_hand')
        self.assertFalse(self.c.started)
        self.assertFalse(self.c.initial_deltas)
        for r in MULTI_ROLES: self.assertIsNone(self.c.pose(r,.7))
        for i in range(8,14): self.step(i)
        self.assertTrue(self.c.started)

    def test_one_common_timestamp_for_all_four_no_extrapolation(self):
        # Install accepted anchors but give the last camera disjoint history.
        for r in MULTI_ROLES:
            for i,t in enumerate((.2,.3) if r=='right_hand' else (0.,.1)):
                self.c.ingest(r,self.packet(r,i,stamp=t),.3)
            self.c.devices[r].anchor=self.anchors[r].copy()
            self.c.devices[r].aligned_at=0.
        self.assertIsNone(self.c.relative(.3))
        self.assertFalse(self.c.started)
        for r in ('chest','head','left_hand'):
            self.c.ingest(r,self.packet(r,3,stamp=.2),.3)
        stamp,relatives=self.c.relative(.3)
        self.assertAlmostEqual(stamp,.2)
        self.assertEqual(len(relatives),3)
        self.assertTrue(self.c.started)

    def test_initial_differences_freeze_after_moving_and_removing_board(self):
        self.initialize()
        original=json.dumps(self.c.initial_deltas,sort_keys=True)
        initial={r:t.copy() for r,t in self.c.initial_poses.items()}
        for i,r in enumerate(MULTI_ROLES):
            self.poses[r][:3,3]+=[i*.1,.2,.05]
            p=self.packet(r,6)
            self.c.ingest(r,p,.6)
            self.c.observe_board(r,p,None,.6)
        self.c.relative(.6)
        self.assertEqual(json.dumps(self.c.initial_deltas,sort_keys=True),original)
        for r in MULTI_ROLES:
            np.testing.assert_array_equal(self.c.initial_poses[r],initial[r])
            np.testing.assert_allclose(self.c.pose(r,.6),self.c.shared_from_board @ self.poses[r],atol=1e-9)
        self.assertEqual(sum(e['event']=='started' for e in self.c.events),1)

    def test_one_lost_stale_or_reset_camera_does_not_hide_the_other_three(self):
        self.initialize()
        fixed=copy.deepcopy(self.c.initial_deltas)
        for r in MULTI_ROLES:
            self.c.ingest(r,self.packet(r,6,valid=r!='left_hand'),.6)
        self.assertIsNone(self.c.pose('left_hand',.6))
        for r in ('chest','head','right_hand'): self.assertIsNotNone(self.c.pose(r,.6))
        old=self.packet('head',6)
        self.c.ingest('head',self.packet('head',7,map_id='reset'),.7)
        self.c.observe_board('head',old,measured(self.poses['head']),.7)
        self.assertIsNone(self.c.devices['head'].anchor)
        self.assertFalse(self.c.devices['head'].window.samples)
        self.assertIsNotNone(self.c.pose('right_hand',.7))
        self.assertEqual(self.c.initial_deltas,fixed)
        self.c.devices['right_hand'].disconnected=True
        self.assertIsNone(self.c.pose('right_hand',.7))
        self.assertIsNotNone(self.c.pose('chest',.7))

    def test_chest_reset_preserves_common_z_and_initial_values(self):
        self.initialize()
        frame=self.c.shared_from_board.copy()
        deltas=copy.deepcopy(self.c.initial_deltas)
        self.anchors['chest']=transform((.8,.2,-.6),(2.,-3.,.4))
        for i in range(6,13):
            for r in MULTI_ROLES:
                self.c.ingest(r,self.packet(r,i,map_id='new' if r=='chest' else '0'),i*.1)
            for r in MULTI_ROLES:
                p=self.c.devices[r].packet
                self.c.observe_board(r,p,measured(self.poses[r]),i*.1)
            self.c.relative(i*.1)
        np.testing.assert_array_equal(self.c.shared_from_board,frame)
        self.assertEqual(self.c.initial_deltas,deltas)
        for r in MULTI_ROLES:
            np.testing.assert_allclose(self.c.pose(r,1.2),frame @ self.poses[r],atol=1e-9)

    def test_recording_snapshots_all_roles_and_initial_differences(self):
        self.initialize()
        with tempfile.TemporaryDirectory() as root:
            args=SimpleNamespace(record=True,session=str(Path(root)/'session'),demo=True)
            recording=Recording(self.cfg,self.devices,args)
            for r in MULTI_ROLES:
                packet={**self.packet(r,5),'host_monotonic':.5}
                recording.pose(r,packet,self.c.pose(r,.5),self.c.board_pose(r,.5))
                recording.demo_capture(r,packet,(np.zeros((16,24,3),np.uint8),)*2)
            recording.finish(self.c,{r:{} for r in MULTI_ROLES})
            session=Path(args.session)
            metadata=json.loads((session/'session.json').read_text())
            self.assertTrue(metadata['synthetic_only'])
            self.assertEqual(metadata['initial_deltas'],self.c.initial_deltas)
            self.assertEqual(set(metadata['initial_T_reference_camera']),{'head','left_hand','right_hand'})
            self.assertEqual(set(json.loads((session/'meta/alignment.json').read_text())),set(MULTI_ROLES))
            from ego2.multi_vio.replay import Reader
            row=list(Reader(session).frames())[0][2]
            for r in MULTI_ROLES: self.assertTrue(row[f'observation.{r}.shared_pose_valid'])
            self.assertFalse((session/'_work').exists())
            for r in MULTI_ROLES: self.assertTrue((session/'calibration'/r/'camera.yaml').is_file())

    def test_blueprint_and_viewpoint_contain_all_devices(self):
        self.initialize()
        blueprint=build_blueprint(self.cfg)
        left,right=blueprint.root_container.contents
        self.assertEqual(len(right.contents),4)
        self.assertEqual([len(row.contents) for row in right.contents],[2]*4)
        self.assertEqual(len(left.contents),2)
        self.assertIn('Δp/m',left.contents[1].name)
        viewer=Visualizer.__new__(Visualizer)
        viewer.cfg=self.cfg; viewer.blueprint=blueprint
        viewer.eye_initialized=False; viewer.recording=MagicMock()
        viewer.initialize_poses({})
        viewer.initialize_poses({'chest':self.c.initial_poses['chest']})
        viewer.recording.send_blueprint.assert_not_called()
        viewer.initialize_poses(self.c.initial_poses)
        viewer.initialize_poses(self.c.initial_poses)
        viewer.recording.send_blueprint.assert_called_once()
        eye=initial_multi_eye_geometry(self.c.initial_poses,self.cfg['viewer'])
        direction=eye['look_target']-eye['position']
        np.testing.assert_allclose(direction/np.linalg.norm(direction),self.c.initial_poses['chest'][:3,2],atol=1e-9)
        np.testing.assert_allclose(eye['look_target'],np.mean([t[:3,3] for t in self.c.initial_poses.values()],axis=0))
        text=relative_document(next(e for e in self.c.events if e['event']=='started'),MULTI_ROLES)
        self.assertEqual(len(text.splitlines()),5)  # fences + three data rows
        self.assertNotIn('LOCKED',text)
        with tempfile.TemporaryDirectory() as root:
            blueprint.save('four-camera-test',str(Path(root)/'layout.rbl'))

    def test_hidden_board_labels_leave_both_display_images_and_inputs_unchanged(self):
        viewer=Visualizer.__new__(Visualizer)
        viewer.cfg=self.cfg; viewer.devices=self.devices; viewer.recording=MagicMock()
        pixels=np.array([[80.,80.],[160.,80.],[160.,160.],[80.,160.]])
        obs=SimpleNamespace(camera='cam0',marker_id=12,reason='reprojection_outlier',pose_reason='ok',
                            pixels=pixels,display_pixels=None,outline=np.r_[pixels,pixels[:1]],display_outline=None)
        track=SimpleNamespace(display_pose=None,status='reprojection_outlier')
        images=tuple(np.full((240,320,3),65,np.uint8) for _ in range(2))
        for image in images: image.flags.writeable=False
        captured=[]
        def image_log(rgb):
            captured.append(rgb.copy())
            return SimpleNamespace(compress=lambda **kwargs:MagicMock())
        with patch('ego2.multi_vio.visualizer.rr.Image',side_effect=image_log):
            viewer.images('head',dict(capture_time=1.),images,[obs],track,'RUNNING',20.)
        for image in images: self.assertTrue(np.all(image==65))
        self.assertTrue(np.all(captured[0]==65))
        self.assertTrue(np.all(captured[1]==65))
        np.testing.assert_array_equal(obs.pixels,pixels)


class MappingTests(unittest.TestCase):
    def raw(self):
        cfg=yaml.safe_load(MULTI_CONFIG.read_text())
        for key in ('aprilgrid_config','orb_source','vocabulary'):
            cfg[key]=str((MULTI_CONFIG.parent/cfg[key]).resolve())
        return cfg

    def load(self,cfg,*,demo=False):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'config.yaml'
            path.write_text(yaml.safe_dump(cfg))
            return load_multi_config(path,demo=demo)

    def test_no_selected_camera_is_rejected_without_loading_any_camera(self):
        cfg=self.raw(); cfg['devices']['chest']['sn']=None
        with patch('ego2.multi_vio.config.load_camera') as camera:
            with self.assertRaisesRegex(ValueError,'至少选择一台'):
                self.load(cfg)
            camera.assert_not_called()

    def test_unassigned_optional_roles_are_disabled_in_live_mode(self):
        cfg,devices=self.load(self.raw())
        self.assertEqual(tuple(devices),('chest',))
        self.assertEqual(cfg['viewer']['row_order'],['chest'])
        self.assertFalse(cfg['viewer']['initial_pose_panel'])

    def test_duplicate_serials_bad_role_order_and_missing_profile_rejected(self):
        cfg=self.raw()
        for r,sn in zip(MULTI_ROLES,('s200100','s200097','s200032','no_such_unit')):
            cfg['devices'][r]['sn']=sn
        with self.assertRaisesRegex(ValueError,'right_hand.*No camera profile'):
            self.load(cfg)
        cfg['devices']['right_hand']['sn']='s200100'
        with self.assertRaisesRegex(ValueError,'different camera SN'):
            self.load(cfg)
        cfg['devices']['right_hand']['sn']='no_such_unit'
        cfg['viewer']['row_order']=['head','chest','left_hand','left_hand']
        with self.assertRaisesRegex(ValueError,'row_order'):
            self.load(cfg)

    def test_role_swap_changes_profile_binding_independent_of_viewer_order(self):
        cfg=self.raw()
        cfg['devices']['chest']['sn']='s200032'
        cfg['devices']['head']['sn']='s200100'
        cfg['devices']['left_hand']['sn']='s200097'
        cfg['devices']['right_hand']['sn']=None
        cfg['viewer']['row_order']=['right_hand','left_hand','head','chest']
        actual,devices=self.load(cfg,demo=True)
        self.assertEqual(devices['chest']['native']['serial'],'s200032')
        self.assertIn('/s200100/',devices['head']['native']['calibration'])
        self.assertIn('/s200097/',devices['left_hand']['native']['calibration'])
        self.assertIsNone(actual['devices']['right_hand']['sn'])
        self.assertEqual(actual['viewer']['row_order'],cfg['viewer']['row_order'])

    def test_all_devices_discovered_before_any_backend_starts(self):
        devices={r:dict(native=dict(serial=f'sn_{r}')) for r in MULTI_ROLES}
        def discover(sn):
            if sn.endswith('head'): raise RuntimeError('missing')
            return '/dev/'+sn
        with patch('ego2.common.sources.discover_cyper',side_effect=discover) as probe:
            with self.assertRaisesRegex(RuntimeError,'head.*missing'):
                check_devices(devices)
            self.assertEqual(probe.call_count,4)
        with patch('ego2.common.sources.discover_cyper',return_value='/dev/video0'):
            with self.assertRaisesRegex(RuntimeError,'same video node'):
                check_devices(devices)

    def test_relative_rotation_gimbal_lock_is_finite(self):
        for sign in (-1,1):
            target=transform((0,sign*np.pi/2,0),(1,2,3))
            d=relative_delta(np.eye(4),target)
            self.assertTrue(np.isfinite(d['delta_rpy_deg']).all())
            np.testing.assert_allclose(d['delta_rpy_deg'],[0,sign*90,0],atol=1e-8)


if __name__ == '__main__': unittest.main()
