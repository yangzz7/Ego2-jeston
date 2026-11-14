import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from ego2.common.config import read_yaml
from ego2.reconstruction.calibration import load_config, calibration, write_orb_settings
from ego2.reconstruction.depth import StereoDepth, disparity_depth, local_points
from ego2.reconstruction.session import DenseMap, eligible_keyframes


class CalibrationTests(unittest.TestCase):
    def test_generated_orb_settings_use_actual_resolution_and_transform_direction(self):
        cfg=load_config()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"orb.yaml"
            shift=write_orb_settings(cfg,path)
            fs=cv2.FileStorage(str(path),cv2.FILE_STORAGE_READ)
            try:
                self.assertEqual(fs.getNode("Camera.width").real(),1280)
                self.assertEqual(fs.getNode("Camera.height").real(),1024)
                self.assertEqual(fs.getNode("Camera.lappingEnd").real(),1279)
                self.assertEqual(fs.getNode("Camera.RGB").real(),0)
                raw=read_yaml(cfg["calibration"])
                np.testing.assert_allclose(fs.getNode("Tbc").mat()@raw["cam1"]["T_cam_imu"],np.eye(4),atol=1e-7)
                np.testing.assert_allclose(fs.getNode("Tlr").mat(),np.array(raw["cam1"]["T_cn_cnm1"])[:3],atol=1e-7)
                self.assertEqual(shift,raw["cam1"]["timeshift_cam_imu"])
                self.assertAlmostEqual(fs.getNode("IMU.NoiseAcc").real(),.0015177735193787072)
            finally:
                fs.release()

    def test_rectified_baseline_and_epipolar_geometry(self):
        cfg=load_config(); depth=StereoDepth(cfg)
        l,r,T=calibration(cfg)
        xyz=np.array([[-.2,.1,1],[.3,-.2,2],[.0,.3,.7]],np.float64)
        pixels=[]
        for cam,points in ((l,xyz),(r,xyz@T[:3,:3].T+T[:3,3])):
            p,_=cv2.fisheye.projectPoints(points.reshape(1,-1,3),np.zeros(3),np.zeros(3),cam.K,cam.D)
            pixels.append(p)
        R1,R2,*_=cv2.fisheye.stereoRectify(l.K,l.D,r.K,r.D,l.size,T[:3,:3],T[:3,3],flags=cv2.CALIB_ZERO_DISPARITY)
        a=cv2.fisheye.undistortPoints(pixels[0],l.K,l.D,R=R1,P=depth.K).reshape(-1,2)
        b=cv2.fisheye.undistortPoints(pixels[1],r.K,r.D,R=R2,P=depth.K).reshape(-1,2)
        np.testing.assert_allclose(a[:,1],b[:,1],atol=1e-7)
        np.testing.assert_allclose(depth.K[0,0]*depth.baseline/(a[:,0]-b[:,0]),(xyz@R1.T)[:,2],atol=1e-7)


class DepthTests(unittest.TestCase):
    def test_lr_rejection_and_metric_range(self):
        cfg=dict(lr_threshold=1,min_m=.3,max_m=3)
        dl=np.full((2,40),5,np.float32); dr=-dl.copy(); valid=np.ones(dl.shape,bool)
        dr[0,10]=0
        depth=disparity_depth(dl,dr,valid,valid,100,.06,cfg)
        self.assertAlmostEqual(depth[1,20],1.2,places=5)
        self.assertEqual(depth[0,15],0)
        self.assertTrue((depth[:,:5]==0).all())
        self.assertTrue((disparity_depth(dl*10,dr,valid,valid,100,.06,cfg)==0).all())

    def test_rectified_points_return_to_native_camera_and_rgb(self):
        angle=.2
        R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        points,colors=local_points(np.ones((1,1),np.float32),np.array([[[10,20,30]]],np.uint8),np.eye(3),R,1,.001)
        np.testing.assert_allclose(points[0],R.T@[0,0,1],atol=1e-7)
        np.testing.assert_array_equal(colors,[[30,20,10]])

    def test_real_fisheye_calibration_synthetic_plane_metric_depth(self):
        from ego2.reconstruction.demo import PlaneScene
        cfg=load_config(); stereo=StereoDepth(cfg)
        a,b=PlaneScene(cfg).render(np.eye(4))
        depth,color=stereo.compute(a,b)
        points,_=stereo.points(depth,color)
        self.assertGreater(np.count_nonzero(depth)/depth.size,.4)
        self.assertGreater(len(points),10000)
        self.assertLess(abs(float(np.median(points[:,2]))-1.3),.035)
        self.assertLess(float(np.percentile(np.abs(points[:,2]-1.3),90)),.06)


class MapTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.viewer=Mock()
        self.dense=DenseMap(load_config(),self.root,self.viewer)
        depth=np.ones((512,640),np.float32)
        color=np.full((512,640,3),100,np.uint8)
        np.savez(self.root/"depth/1000000.npz",depth=depth,color=color)
        self.snapshot=dict(version=2,state=2,active_map="0_0",mode="stereo-inertial",final=False,
            maps=[dict(id="0_0",imu_initialized=True,imu_initialization_complete=True,revision=0,keyframes=[dict(id=1,timestamp=1,
            image_key="1000000",image_available=True,T_world_camera=np.eye(4).tolist())])])

    def tearDown(self):
        self.tmp.cleanup()

    def test_loop_update_moves_chunk_without_reintegrating(self):
        self.dense.update(self.snapshot,0)
        self.snapshot["maps"][0]["keyframes"][0]["T_world_camera"][0][3]=.5
        self.dense.update(self.snapshot,1)
        self.assertEqual(self.viewer.chunk.call_count,1)
        self.assertEqual(self.viewer.pose.call_count,1)
        self.assertAlmostEqual(self.viewer.pose.call_args.args[2][0,3],.5)

    def test_reset_clears_old_coordinate_frame(self):
        self.dense.update(self.snapshot,0)
        self.snapshot["active_map"]="0_1"
        self.snapshot["maps"]=[]
        self.dense.update(self.snapshot,1)
        self.viewer.remove_chunk.assert_called_once_with("0_0","1000000")
        self.assertFalse(self.dense.live)

    def test_map_merge_reparents_points(self):
        self.dense.update(self.snapshot,0)
        self.snapshot["active_map"]="3_0"
        self.snapshot["maps"][0]["id"]="3_0"
        self.dense.update(self.snapshot,1)
        self.viewer.remove_chunk.assert_called_once_with("0_0","1000000")
        self.assertEqual(self.viewer.chunk.call_args.args[0],"3_0")

    def test_tracking_loss_pauses_new_chunks(self):
        self.snapshot["state"]=3
        self.dense.update(self.snapshot,0)
        self.viewer.chunk.assert_not_called()

    def test_uninitialized_imu_and_missing_images_are_not_fused(self):
        self.snapshot["maps"][0]["imu_initialized"]=False
        self.assertEqual(eligible_keyframes(self.snapshot),{})
        self.snapshot["maps"][0]["imu_initialized"]=True
        self.snapshot["maps"][0]["keyframes"][0]["image_available"]=False
        self.assertEqual(eligible_keyframes(self.snapshot),{})

    def test_independent_maps_are_not_overlaid(self):
        self.snapshot["active_map"]="1_0"
        self.dense.update(self.snapshot,0)
        self.viewer.chunk.assert_not_called()

    def test_initial_ba_must_finish_before_any_depth_or_points(self):
        self.snapshot["maps"][0]["imu_initialization_complete"]=False
        original=self.dense.depth
        self.dense.depth=Mock(wraps=original)
        self.dense.update(self.snapshot,0)
        self.assertFalse(self.dense.ready)
        self.dense.depth.assert_not_called()
        self.viewer.chunk.assert_not_called()
        self.assertEqual(eligible_keyframes(self.snapshot),{})
        self.assertEqual(json.loads((self.root/"dense_status.json").read_text())["reconstruction_state"],"waiting_for_vio")
        self.snapshot["maps"][0]["imu_initialization_complete"]=True
        self.dense.update(self.snapshot,1)
        self.assertTrue(self.dense.ready)
        self.viewer.chunk.assert_called_once()

    def test_reset_to_initializing_clears_even_cached_point_clouds(self):
        self.dense.update(self.snapshot,0)
        self.snapshot["maps"][0]["imu_initialization_complete"]=False
        self.dense.depth=Mock(side_effect=AssertionError("depth before initialization"))
        self.dense.update(self.snapshot,1)
        self.viewer.remove_chunk.assert_called_once_with("0_0","1000000")
        self.assertFalse(self.dense.live)
        self.assertFalse(self.dense.ready)
        self.assertEqual(json.loads((self.root/"dense_status.json").read_text())["live_points"],0)

    def test_final_flush_cannot_bypass_initialization(self):
        self.snapshot["maps"][0]["imu_initialization_complete"]=False
        self.snapshot["final"]=True
        self.dense.depth=Mock(side_effect=AssertionError("depth before initialization"))
        self.dense.update(self.snapshot,0,budget=None)
        self.dense.depth.assert_not_called()
        self.viewer.chunk.assert_not_called()

    def test_unknown_completion_in_new_protocol_is_not_ready(self):
        del self.snapshot["maps"][0]["imu_initialization_complete"]
        self.dense.update(self.snapshot,0)
        self.assertFalse(self.dense.ready)
        self.viewer.chunk.assert_not_called()

    def test_offline_export_respects_completion_for_every_map(self):
        pending=copy.deepcopy(self.snapshot["maps"][0])
        pending.update(id="1_0",imu_initialization_complete=False)
        pending["keyframes"][0]["image_key"]="2000000"
        self.snapshot["maps"].append(pending)
        self.snapshot["active_map"]="1_0"
        self.dense.depth=Mock()
        self.dense.update(self.snapshot,0)
        self.dense.depth.assert_not_called()
        self.dense.prepare_export(self.snapshot)
        self.dense.depth.assert_called_once_with("1000000")

    def test_legacy_initialization_fallback_is_only_for_offline_export(self):
        self.snapshot["version"]=1
        del self.snapshot["maps"][0]["imu_initialization_complete"]
        self.dense.depth=Mock()
        self.dense.update(self.snapshot,0)
        self.dense.depth.assert_not_called()
        self.assertFalse(self.dense.ready)
        self.dense.prepare_export(self.snapshot)
        self.dense.depth.assert_called_once_with("1000000")


if __name__=="__main__":
    unittest.main()
