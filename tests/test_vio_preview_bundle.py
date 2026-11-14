"""Cross-language atomic preview snapshots and stable enhancement failures."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import MagicMock

import zmq

import numpy as np

from ego2.common.display_isp import DisplayIsp
from ego2.common.paths import BUILD
from ego2.multi_vio.live_preview import TrajectoryPreview
from ego2.multi_vio.transport import Backend, RingReader
from tests.test_live_display_isp import FakeIsp
from tests.test_vio_live_preview import packet


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'frames.bin'
        self.executable=os.environ.get('EGO2_PREVIEW_RING_TEST',str(BUILD/'bin/test_vio_preview_bundle'))

    def make(self, mode='normal'):
        subprocess.run([self.executable,str(self.path),mode],check=True,capture_output=True)
        reader=RingReader(self.path,version=2)
        self.addCleanup(reader.close)
        return reader

    def check_bundle(self, bundle, n):
        images,raw=bundle
        value=n%199
        for index,image in enumerate(images):
            np.testing.assert_array_equal(image,np.broadcast_to(
                np.array([value+3*index+i for i in range(3)],np.uint8),image.shape))
        np.testing.assert_array_equal(raw[:-1],np.broadcast_to(
            np.array([value+6,value+7],np.uint8),raw[:-1].shape))
        np.testing.assert_array_equal(raw[-1,0],[2,7])

    def test_cpp_snapshot_preserves_both_eyes_raw_metadata_and_rejects_overwrite(self):
        reader=self.make()
        self.check_bundle(reader.read_bundle(dict(slot=1,frame_id=4)),4)
        self.assertIsNone(reader.read_bundle(dict(slot=1,frame_id=1)))
        with self.assertRaisesRegex(ValueError,'size'):
            RingReader(self.path)  # A stale consumer must not parse v2 as v1.

    def test_absent_raw_clears_presence_in_reused_slot_but_keeps_rgb(self):
        reader=self.make('missing')
        images,raw=reader.read_bundle(dict(slot=1,frame_id=4))
        self.assertIsNone(raw)
        self.assertEqual(images[0][0,0].tolist(),[4,5,6])
        self.assertEqual(images[1][0,0].tolist(),[7,8,9])

    def test_concurrent_overwrite_never_mixes_raw_and_rgb_or_changes_owned_snapshots(self):
        process=subprocess.Popen([self.executable,str(self.path),'stress'],stdout=subprocess.PIPE,text=True)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        reader=None; snapshots=[]
        try:
            for line in process.stdout:
                n=int(line)
                if reader is None: reader=RingReader(self.path,version=2)
                bundle=reader.read_bundle(dict(slot=n%3,frame_id=n))
                if bundle is not None: snapshots.append((n,bundle))
                # Bound retained test memory, keeping an early owned copy too.
                if len(snapshots)>4: snapshots.pop(1)
            self.assertEqual(process.wait(timeout=10),0)
            self.assertGreaterEqual(len(snapshots),1)
            for n,bundle in snapshots: self.check_bundle(bundle,n)
        finally:
            if reader: reader.close()
            process.stdout.close()
            if process.poll() is None: process.kill(); process.wait()

    def test_backend_only_returns_the_snapshot_for_the_received_frame(self):
        backend=Backend.__new__(Backend)
        backend.display_raw=True
        backend.display_snapshot=(4,np.full((8,8,2),44,np.uint8))
        self.assertIs(backend.read_display_raw({'frame_id':4}),backend.display_snapshot[1])
        self.assertIsNone(backend.read_display_raw({'frame_id':7}))
        backend.display_snapshot=None
        self.assertIsNone(backend.read_display_raw({'frame_id':4}))

    def test_backend_receive_owns_raw_before_the_preview_or_board_worker_runs(self):
        reader=self.make()
        backend=Backend.__new__(Backend)
        backend.reader=reader; backend.display_snapshot=None; backend.display_raw=True
        backend.socket=MagicMock(); backend.clock=MagicMock()
        backend.clock.update.return_value=(.4,0.)
        backend.cfg={'serial':'test-camera'}; backend.last_frame=-1; backend.overwritten=0
        p=dict(slot=1,frame_id=4,serial='test-camera',protocol=1,cameras=['cam0','cam1'],
               timestamp=.4,host_monotonic=1.,valid=False)
        backend.socket.recv_json.side_effect=[p,zmq.Again()]
        received,images=backend.receive()
        raw=backend.read_display_raw(received)
        reader.close()
        self.check_bundle((images,raw),4)
        backend.socket.recv_json.side_effect=[zmq.Again()]
        self.assertIsNone(backend.receive())
        self.assertIsNone(backend.read_display_raw(received))


class StablePreviewTests(unittest.TestCase):
    def test_missing_raw_and_isp_exception_skip_images_and_matching_annotations(self):
        class Processor(FakeIsp):
            def process_transport(self, raw):
                if raw[0,0,0]==4: raise RuntimeError('synthetic ISP failure')
                return super().process_transport(raw)
        worker=DisplayIsp(Processor(),max_fps=1000)
        published=[]; annotated=[]
        def publish(role,p,images,status):
            published.append((p,images))
        preview=TrajectoryPreview(('head',),{'head':worker},85,1000,publish,
                                  lambda role,p,*a: annotated.append(p['frame_id']))
        original=tuple(np.full((24,32,3),v,np.uint8) for v in (11,22))
        try:
            with redirect_stdout(io.StringIO()) as messages:
                for n in (1,2,3,4,5):
                    raw=None if n==2 else np.full((24,32,2),n,np.uint8)
                    preview.board('head',packet(n),[],None,'RUNNING',1)
                    preview.submit('head',packet(n),original,'RUNNING',raw)
                    # Observe completion so the latest-only queue cannot hide
                    # the deliberately failing frames in this recovery test.
                    deadline=time.monotonic()+3
                    while worker.rendered+worker.skipped<n and time.monotonic()<deadline:
                        time.sleep(.002)
                    self.assertEqual(worker.rendered+worker.skipped,n)
                    self.assertEqual(worker.fallback,0)
                    self.assertEqual(worker.skipped,sum(i<=n for i in (2,4)))
            self.assertIn('保留上一帧及原时间戳',messages.getvalue())
        finally: preview.close()
        self.assertEqual([p['frame_id'] for p,_ in published],[1,3,5])
        self.assertEqual([p['capture_time'] for p,_ in published],[n*.1 for n in (1,3,5)])
        self.assertEqual(annotated,[1,3,5])
        self.assertFalse(preview.pending['head'])
        for image,value in zip(original,(11,22)): np.testing.assert_array_equal(image,value)

    def test_default_other_modes_keep_existing_same_frame_fallback(self):
        worker=DisplayIsp(FakeIsp(),max_fps=1000)
        original=(np.zeros((8,8,3),np.uint8),)*2
        shown=[]
        worker.submit(None,original,shown.append)
        worker.close()
        self.assertIs(shown[0],original)
        self.assertEqual(worker.fallback,1)


if __name__=='__main__': unittest.main()
