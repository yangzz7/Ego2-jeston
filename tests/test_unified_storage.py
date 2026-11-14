"""V3 index/video round trips, asynchronous pairing and portable replay."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pyarrow.parquet as pq

from ego2.multi_vio import storage
from ego2.multi_vio.dataset import Writer, sample, assemble, save_imu, trajectory_sources
from ego2.multi_vio.replay import Reader, export_dataset
from ego2.multi_vio.__main__ import main


def frame(t,seq,value=40):
    images={'cam0':np.full((32,40,3),value,np.uint8),'cam1':np.full((32,40,3),value+40,np.uint8)}
    return sample(t,images,100+t,200+t,seq)


def metadata(roles):
    return dict(devices={r:'test-'+r for r in roles},config=dict(mode='gesture',devices={r:dict(sn='test-'+r) for r in roles},
        viewer=dict(row_order=list(roles),memory_limit='100MiB')))


class StorageTests(unittest.TestCase):
    def test_directory_clock_collision_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d,patch.object(storage,'OUTPUT',Path(d)):
            a=storage.new_session('vio');b=storage.new_session('gesture')
            self.assertEqual(set(p.name for p in Path(d).iterdir()),set(storage.FOLDERS.values()))
            self.assertEqual(a.parent.name,'trajectory')
            self.assertRegex(a.name,r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}$')
            with self.assertRaises(FileExistsError):storage.new_session('gesture',b)

    def test_v3_video_index_offsets_across_episodes_are_lossless(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);w=Writer(root,['head'],10,'gesture',episode_frames=2)
            for n in range(5):w.append(n*.1,{'head':frame(n*.1,n,20+n)})
            w.close();storage.write_json(root/'meta/ego2.json',dict(schema_version=2,mode='gesture',**metadata(['head'])))
            reader=Reader(root);frames=list(reader.frames())
            self.assertEqual(reader.info['codebase_version'],'v3.0')
            self.assertEqual(reader.info['total_episodes'],3)
            self.assertEqual([f[2]['index'] for f in frames],list(range(5)))
            for n,(_,images,row) in enumerate(frames):
                self.assertTrue(np.all(images['head']['cam0']==20+n))
                self.assertTrue(np.all(images['head']['cam1']==60+n))
                self.assertEqual(row['episode_index'],n//2)
            # Corrupt the episode index: decoding must not silently pair wrong frames.
            ep=root/'meta/episodes/chunk-000/file-001.parquet'
            values=pq.read_table(ep).to_pylist();values[0]['dataset_from_index']=99
            import pyarrow as pa
            pq.write_table(pa.Table.from_pylist(values),ep)
            with self.assertRaisesRegex(ValueError,'episode index'):list(Reader(root).frames())

    def test_async_missing_frame_is_black_and_invalid_not_stale_pose(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            a=[frame(t,n) for n,t in enumerate([0.,.1,.2])]
            b=[frame(.01,20,150)]
            b[0].update(shared_pose_valid=True,shared_pose=np.eye(4).reshape(-1).tolist())
            assemble(root,'gesture',{'chest':iter(a),'head':iter(b)},10,metadata(['chest','head']))
            saved=list(Reader(root).frames())
            self.assertTrue(saved[0][2]['observation.head.image_valid'])
            self.assertAlmostEqual(saved[0][2]['observation.head.time_offset_s'],.01)
            self.assertFalse(saved[1][2]['observation.head.image_valid'])
            self.assertFalse(saved[1][2]['observation.head.shared_pose_valid'])
            self.assertTrue(np.all(saved[1][1]['head']['cam0']==0))
            path=export_dataset(root,root/'export.rrd')
            self.assertGreater(path.stat().st_size,1000)
            with self.assertRaises(FileExistsError):export_dataset(root,path)

    def test_hand_overlay_stays_on_correct_camera_after_dataset_replay(self):
        from dataclasses import asdict
        from tests.test_gesture_dataset import make_hand
        from rerun.experimental import RrdReader
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);s=frame(0.,0)
            s['inference_valid']=True
            s['result']={'hands':{'cam1':[asdict(make_hand())]}}
            assemble(root,'gesture',{'head':iter([s])},30,metadata(['head']))
            saved=list(Reader(root).frames())[0][2]
            self.assertEqual(saved['observation.head.hands.cam1.present'],[True,False])
            self.assertEqual(saved['observation.head.hands.cam0.present'],[False,False])
            rrd=export_dataset(root,root/'replay.rrd')
            paths={c.entity_path for c in RrdReader(str(rrd)).stream()}
            self.assertIn('/cameras/head/cam1/hands/7/points',paths)
            self.assertNotIn('/cameras/head/cam0/hands/7/points',paths)

    def test_map_reset_cannot_reuse_old_shared_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);directory=work/'head/native';directory.mkdir(parents=True)
            states=[dict(role='head',frame_id=0,map_id='A',epoch=1,status='RUNNING',T_shared_local=np.eye(4).tolist())]
            (work/'poses.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in states))
            (work/'boards.jsonl').touch()
            packets=[]
            for n,map_id in enumerate(['A','B']):
                images={}
                for cam in ('cam0','cam1'):
                    name=f'{n}_{cam}.png';images[cam]=name
                    cv2.imwrite(str(directory/name),np.full((16,16,3),n,np.uint8))
                packets.append(dict(frame_id=n,map_id=map_id,timestamp=n,host_monotonic=n,
                                    images=images,valid=True,T_world_camera=np.eye(4).tolist()))
            (directory/'frames.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in packets))
            samples=list(trajectory_sources(work,['head'],0,True)['head'])
            self.assertTrue(samples[0]['shared_pose_valid'])
            self.assertTrue(samples[1]['local_pose_valid'])
            self.assertFalse(samples[1]['shared_pose_valid'])

    def test_imu_dedup_preserves_native_rate_and_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'imu.parquet'
            samples=[dict(timestamp_seconds=t,acceleration_m_s2=[0,0,9.8],angular_velocity_rad_s=[0,0,0])
                     for t in [1.,1.005,1.005,1.01,1.2]]
            save_imu(path,samples)
            self.assertEqual(pq.read_table(path)['timestamp_seconds'].to_pylist(),[1.,1.005,1.01,1.2])

    def test_replay_does_not_discover_hardware_or_load_live_config(self):
        with patch('ego2.multi_vio.replay.replay') as replay,\
             patch('ego2.multi_vio.profiles.prepare_camera_profiles',side_effect=AssertionError),\
             patch('ego2.multi_vio.__main__.load_multi_config',side_effect=AssertionError):
            self.assertEqual(main(['replay','--session','/example','--headless','--save','/tmp/example.rrd']),0)
            replay.assert_called_once()

    def test_broken_rrd_keeps_work_and_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'_work';work.mkdir();(work/'important').write_text('retained')
            bad=root/'broken.rrd';bad.write_bytes(b'not-rerun')
            with self.assertRaises(Exception):storage.finish_replay(root,bad,work)
            self.assertTrue((work/'important').exists())


if __name__=='__main__':unittest.main()
