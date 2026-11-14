"""Stop/flush/restart boundaries while capture and preview remain alive."""
from concurrent.futures import Future
from io import StringIO
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import pyarrow.parquet as pq
from ego2.multi_vio.manual import ManualController, WorkerControl, WorkerGroup
from ego2.multi_vio.storage import write_json, status


class StopTests(unittest.TestCase):
    def until(self, predicate, step=lambda:None):
        end=time.monotonic()+5
        while not predicate() and time.monotonic()<end:
            step(); time.sleep(.005)
        self.assertTrue(predicate())

    def test_enter_stops_without_panel_background_flush_and_second_recording(self):
        with tempfile.TemporaryDirectory() as directory, patch('subprocess.Popen') as popen:
            gate=threading.Event(); prepare=Mock(); stop=Mock(); stopped=Mock(return_value=False)
            c=ManualController('gesture',SimpleNamespace(session=None),prepare,lambda:True,Mock(),Mock(),
                stdin=StringIO(''),stop=stop,stopped=stopped)
            def finalize():
                root=c.root
                def finish():
                    gate.wait(5);status(root,'complete')
                return finish
            c.finalize=Mock(side_effect=finalize)
            try:
                write_json(c.directory/'request.json',dict(id='one',directory=directory,name='one'))
                c.poll();self.assertTrue(c.active)
                (c.directory/'open-panel').touch();c.poll()
                self.assertTrue(c.stopping);stop.assert_called_once();popen.assert_not_called()
                c.open();c.poll();c.finalize.assert_not_called()
                stopped.return_value=True;c.poll()
                self.assertIsNotNone(c.finish_future)
                before=time.monotonic()
                for _ in range(10):c.open();c.poll()
                self.assertLess(time.monotonic()-before,.5)
                self.assertEqual(len(c.saved),0);c.finalize.assert_called_once()
                gate.set();self.until(lambda:len(c.saved)==1,c.poll)
                self.assertFalse(c.active);self.assertFalse(c.stopping)
                c.poll();prepare.assert_called_once()  # Old panel request is not replayed.
                write_json(c.directory/'request.json',dict(id='two',directory=directory,name='two'))
                c.poll();self.assertTrue(c.active);self.assertEqual(prepare.call_count,2)
                self.assertEqual(c.root.name,'two')
            finally:
                gate.set();c.close()

    def test_failed_finalizer_preserves_incomplete_status_and_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            c=ManualController('aprilgrid',SimpleNamespace(session=None),Mock(),lambda:True,Mock(),Mock(),
                stdin=StringIO(''),stop=Mock(),stopped=lambda:True,
                finalize=lambda:Mock(side_effect=RuntimeError('write failed')))
            try:
                write_json(c.directory/'request.json',dict(id='one',directory=directory,name='one'))
                c.poll();c.open();c.poll()
                self.until(lambda:not c.stopping,c.poll)
                self.assertFalse(c.active);self.assertFalse(c.saved)
                meta=json.loads((c.root/'session.json').read_text())
                self.assertEqual(meta['status'],'incomplete');self.assertIn('write failed',meta['error'])
            finally:c.close()

    def test_worker_acknowledges_after_flush_and_old_token_cannot_finish_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            group=WorkerGroup(Path(directory)/'control',['head','chest'])
            futures={r:Future() for r in group.directories}
            workers={r:WorkerControl(path,finish=lambda r=r:futures[r]) for r,path in group.directories.items()}
            group.prepare(Path(directory)/'one','one')
            for w in workers.values():w.poll()
            group.commit(1.);group.stop(2.)
            for _ in range(2):
                for w in workers.values():w.poll()
            self.assertFalse(group.stopped())
            futures['head'].set_result(None);workers['head'].poll();self.assertFalse(group.stopped())
            futures['chest'].set_result(None);workers['chest'].poll();self.assertTrue(group.stopped())
            group.prepare(Path(directory)/'two','two');self.assertFalse(group.stopped())
            for w in workers.values():w.poll()
            self.assertTrue(group.ready());self.assertFalse(group.stopped())

    def test_gesture_two_recordings_exclude_post_stop_pairs(self):
        from ego2.multi_vio.gesture_recording import LiveGestureRecording
        from tests.test_gesture_dataset import make_pair
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);group=WorkerGroup(root/'control',['head'])
            args=SimpleNamespace(cameras={},dataset_fps=None,task='test',episode_frames=900,min_free_gb=0)
            profile={'input':{'type':'demo'},'channels':[{'camera':'cam0'},{'camera':'cam1'}]}
            worker=LiveGestureRecording(group.directories['head'],profile,args,SimpleNamespace(fps=30))
            try:
                for round,seq in enumerate((3,10),1):
                    target=root/str(round);group.prepare(target,str(round));worker.poll()
                    group.commit(900+seq*.04-.01)
                    for i in (seq,seq+1):
                        p=make_pair(i);worker.capture(p);worker.result(p,1.,[[],[]],[],2.)
                    group.stop(900+(seq+1)*.04+.01)
                    worker.capture(make_pair(seq+2));worker.result(make_pair(seq+2),1.,[[],[]],[],2.)
                    self.until(group.stopped,worker.poll)
                    rows=pq.read_table(next((target/'_work/head/data/data').glob('*/*.parquet'))).to_pylist()
                    self.assertEqual([r['observation.source_sequence'] for r in rows],[seq,seq+1])
                    self.assertIsNone(worker.dataset)
            finally:worker.close()


if __name__=='__main__':unittest.main()
