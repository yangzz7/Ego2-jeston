"""Recording boundaries and panel protocol, independent of connected hardware."""
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import tempfile
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pyarrow.parquet as pq

from ego2.multi_vio import manual
from ego2.multi_vio.storage import write_json


class ManualTests(unittest.TestCase):
    def test_terminal_enter_opens_once_and_does_not_create_a_dataset(self):
        read,write=os.pipe()
        with os.fdopen(read) as terminal:
            process=Mock();process.poll.return_value=None
            with patch.object(manual.subprocess,'Popen',return_value=process) as popen:
                c=manual.ManualController('gesture',SimpleNamespace(session=None),Mock(),Mock(),Mock(),Mock(),stdin=terminal)
                os.write(write,b'\n');c.poll()
                os.write(write,b'\n');c.poll()
                popen.assert_called_once();self.assertIsNone(c.root)
                c.close()
        os.close(write)

    def test_defaults_custom_names_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d, patch.object(manual,'OUTPUT',Path(d)):
            now=datetime(2026,10,30,12,34,56,123456)
            self.assertEqual(manual.destination('vio',now=now),Path(d)/'trajectory/2026-10-30_12-34-56-123456')
            self.assertFalse((Path(d)/'trajectory').exists())
            target=manual.create_destination('gesture',d,'抓取测试 01')
            self.assertEqual(target,Path(d)/'gesture/抓取测试 01')
            (target/'keep').write_text('keep')
            with self.assertRaises(ValueError):manual.create_destination('gesture',d,'抓取测试 01')
            self.assertEqual((target/'keep').read_text(),'keep')
            for name in ('../escape','a/b','a\\b','..','\x00'):
                with self.subTest(name=name),self.assertRaises(ValueError):manual.destination('vio',d,name)
            with self.assertRaises(ValueError):manual.destination('vio','relative','name')

    def test_controller_cancel_is_empty_and_prepare_waits_for_all_workers(self):
        with tempfile.TemporaryDirectory() as d:
            prepare,ready,commit,abort=Mock(),Mock(return_value=False),Mock(),Mock()
            c=manual.ManualController('gesture',SimpleNamespace(session=None),prepare,ready,commit,abort,stdin=StringIO(''))
            c.poll();self.assertIsNone(c.root);prepare.assert_not_called()
            write_json(c.directory/'request.json',dict(id='one',directory=d,name='run'))
            c.poll();self.assertTrue(c.pending);commit.assert_not_called()
            ready.return_value=True
            c.poll();self.assertTrue(c.active);commit.assert_called_once()
            self.assertEqual(json.loads((c.directory/'response.json').read_text())['ok'],True)
            c.poll();commit.assert_called_once();c.close()
            c2=manual.ManualController('gesture',SimpleNamespace(session=None),prepare,ready,commit,abort,stdin=StringIO(''))
            c2.close();self.assertEqual(list((Path(d)/'gesture').iterdir()),[Path(d)/'gesture/run'])

    def test_failed_preparation_keeps_dialog_retryable_and_does_not_commit(self):
        with tempfile.TemporaryDirectory() as d:
            commit,abort=Mock(),Mock()
            c=manual.ManualController('vio',SimpleNamespace(session=None),Mock(),Mock(side_effect=RuntimeError('camera failed')),
                                      commit,abort,stdin=StringIO(''))
            write_json(c.directory/'request.json',dict(id='one',directory=d,name='failed'))
            c.poll();self.assertFalse(c.active);self.assertFalse(c.pending)
            commit.assert_not_called();abort.assert_called_once()
            self.assertIn('camera failed',json.loads((c.directory/'response.json').read_text())['error'])
            self.assertEqual(json.loads((Path(d)/'trajectory/failed/session.json').read_text())['status'],'incomplete')
            c.close()

    def test_gesture_capture_and_inference_exclude_frames_before_commit_cutoff(self):
        from ego2.multi_vio.gesture_recording import LiveGestureRecording
        from tests.test_gesture_dataset import make_pair
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);group=manual.WorkerGroup(root/'control',['head'])
            args=SimpleNamespace(cameras={},dataset_fps=None,task='test',episode_frames=900,min_free_gb=0)
            profile={'input':{'type':'demo'},'channels':[{'camera':'cam0'},{'camera':'cam1'}]}
            worker=LiveGestureRecording(group.directories['head'],profile,args,SimpleNamespace(fps=30))
            worker.capture(make_pair(0));self.assertFalse((root/'session').exists())
            group.prepare(root/'session','one');worker.poll();self.assertTrue(group.ready())
            worker.capture(make_pair(1))
            group.commit(900.1)
            for n in (2,3,4):
                p=make_pair(n);worker.capture(p);worker.result(p,1.,[[],[]],[],2.)
            worker.close()
            data=root/'session/_work/head/data'
            rows=pq.read_table(next((data/'data').glob('*/*.parquet'))).to_pylist()
            self.assertEqual([r['observation.source_sequence'] for r in rows],[3,4])
            self.assertTrue(all(r['observation.inference_valid'] for r in rows))

    def test_aprilgrid_rrd_rejects_queued_pre_record_frames(self):
        import queue
        import rerun as rr
        from ego2.multi_vio.mode_viewer import UnifiedViewer,BatchRecording
        stream=Mock();viewer=UnifiedViewer.__new__(UnifiedViewer);viewer.recording=stream
        q=queue.Queue();batch=BatchRecording(q,'head',100,eye_names=('cam0','cam1'))
        for received in (104.,106.):
            batch.begin(1.,received=received,sequence=int(received))
            batch.log('/head/eye0/rgb',rr.TextDocument('frame'));batch.finish()
        viewer.consume(q.get(),cutoff=105.);stream.log.assert_not_called()
        viewer.consume(q.get(),cutoff=105.);stream.log.assert_called_once()
        self.assertIn(unittest.mock.call('capture_time',duration=1.),stream.set_time.call_args_list)

    def test_reconstruction_snapshot_contains_only_current_visible_map(self):
        from ego2.multi_vio.reconstruction_recording import ReconstructionRecording
        recorder=ReconstructionRecording.__new__(ReconstructionRecording)
        recorder.preview=None;recorder.file=Mock();recorder.start_time=10.;recorder.offset=None
        recorder.end_time=None
        recorder.snapshot_pending=True
        recorder.dense=SimpleNamespace(live={'current':('map2',np.eye(4))},depth=Mock(return_value=(1,2)),
                                       stereo=SimpleNamespace(points=Mock(return_value=(np.zeros((1,3)),np.zeros((1,3))))))
        with patch('ego2.multi_vio.reconstruction_recording.time.monotonic',return_value=9.):recorder.time(100.)
        recorder.file.chunk.assert_not_called()
        with patch('ego2.multi_vio.reconstruction_recording.time.monotonic',return_value=11.):
            recorder.time(101.);recorder.time(102.)
        recorder.file.chunk.assert_called_once()
        self.assertEqual(recorder.file.chunk.call_args.args[:2],('map2','current'))
        self.assertEqual(recorder.offset,101.)


if __name__=='__main__':unittest.main()
