"""Delayed same-frame board observations must align without reviving stale VIO."""
from collections import deque
from concurrent.futures import Future
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ego2.multi_vio.alignment import Coordinator, inverse
from ego2.multi_vio.app import live
from ego2.multi_vio.config import load_multi_config
from tests.test_multi_vio_core import measured, transform


class BoardLatencyTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.devices=load_multi_config(selection={'chest':'s200097'})
        self.cfg['timing']['board_max_age_s']=2.
        self.c=Coordinator(self.cfg,self.devices)
        self.anchor=transform((.2,-.1,.3),(.5,.2,-.1))
        self.state=self.c.devices['chest']

    def packet(self,n,*,valid=True,map_id='0',stamp=None):
        stamp=n*.1 if stamp is None else stamp
        pose=transform((.1,.05*stamp,.03*stamp),(.02*stamp,.1,.8))
        return dict(frame_id=n,timestamp=stamp,capture_time=stamp,host_monotonic=stamp,
                    map_id=map_id,valid=valid,initialization_complete=True,
                    T_world_camera=(inverse(self.anchor)@pose).tolist()),measured(pose)

    def ingest(self,n,**kwargs):
        packet,board=self.packet(n,**kwargs)
        self.c.ingest('chest',packet,packet['capture_time']+.1)
        return packet,board

    def observe(self,pair,now):
        self.c.observe_board('chest',*pair,now)

    def test_one_second_late_moving_camera_aligns_with_20_real_frame_pairs(self):
        pending=deque()
        for n in range(65):
            pair=self.ingest(n)
            now=n*.1+.1
            if n%2==0: pending.append((n*.1+1.,pair))
            while pending and pending[0][0]<=now:
                _,result=pending.popleft()
                self.observe(result,now)
            self.c.relative(now)
        self.assertTrue(self.c.started)
        self.assertGreaterEqual(self.state.window.quality['inliers'],20)
        np.testing.assert_allclose(self.state.anchor,self.anchor,atol=1e-9)
        self.assertIsNotNone(self.c.pose('chest',6.5))
        # The live deadline stays 0.6 s, independently of the board deadline.
        self.assertIsNone(self.c.pose('chest',7.01))

    def test_over_budget_result_keeps_already_accepted_samples(self):
        pairs=[self.ingest(n) for n in range(31)]
        self.observe(pairs[20],3.1)
        self.assertEqual(len(self.state.window.samples),1)
        self.observe(pairs[5],3.11)
        self.assertEqual(self.state.window.reason,'board_processing_too_old')
        self.assertEqual(len(self.state.window.samples),1)
        self.observe(pairs[22],3.12)
        self.assertEqual(len(self.state.window.samples),2)

    def test_matching_uses_saved_source_pose_and_capture_clock(self):
        pairs=[self.ingest(n) for n in range(16)]
        old,board=pairs[5]
        old['T_world_camera']=pairs[-1][0]['T_world_camera']
        self.observe((old,board),1.6)
        np.testing.assert_allclose(self.state.window.samples[0][1],self.anchor,atol=1e-9)
        for n in range(16,31): self.ingest(n)
        old['host_monotonic']=3.1  # A mutated result cannot renew its source age.
        self.observe((old,board),3.1)
        self.assertEqual(self.state.window.reason,'board_processing_too_old')

    def test_result_before_tracking_loss_cannot_align_after_recovery(self):
        old=self.ingest(0)
        self.ingest(1,valid=False)
        current=self.ingest(2)
        self.observe(old,.3)
        self.assertEqual(self.state.window.reason,'board_frame_not_in_current_track')
        self.assertFalse(self.state.window.samples)
        self.observe(current,.3)
        self.assertEqual(len(self.state.window.samples),1)

    def test_result_cannot_cross_map_reset_even_if_old_map_id_returns(self):
        old=self.ingest(0,map_id='a')
        self.ingest(1,map_id='b')
        self.ingest(2,map_id='a')
        self.observe(old,.3)
        self.assertFalse(self.state.window.samples)
        self.assertIsNone(self.state.anchor)

    def test_silent_stream_gap_invalidates_pending_board_frames(self):
        old=self.ingest(0)
        self.ingest(10)
        self.observe(old,1.1)
        self.assertEqual(self.state.window.reason,'board_frame_not_in_current_track')
        self.assertFalse(self.state.window.samples)

    def test_board_sample_gap_still_restarts_collection(self):
        # Explicit short-gap policy stays supported for external profiles.
        self.cfg['alignment']['max_sample_gap_s']=.6
        pairs=[self.ingest(n) for n in range(31)]
        self.observe(pairs[20],3.1)
        self.observe(pairs[30],3.11)
        self.assertEqual(len(self.state.window.samples),1)
        self.assertEqual(self.state.window.samples[0][0],3.)

    def test_wrong_frame_identity_does_not_use_current_pose(self):
        packet,board=self.ingest(0)
        wrong=dict(packet,frame_id=99)
        self.observe((wrong,board),.1)
        self.assertFalse(self.state.window.samples)

    def test_pending_history_is_bounded(self):
        old=self.ingest(0)
        for n in range(1,160): self.ingest(n,stamp=n*.005)
        self.assertLessEqual(len(self.state.board_packets),120)
        self.observe(old,.9)
        self.assertFalse(self.state.window.samples)


class BoardSchedulingTests(unittest.TestCase):
    def test_receive_fresh_vio_before_evaluating_finished_board_result(self):
        output=MagicMock();output.counts={'chest':{'frames':0}}
        output.viewer=None;output.recording.session=None
        events=[]
        def ingest(role,packet,now):
            output.counts[role]['frames']+=1;events.append(('vio',packet['frame_id']))
        output.ingest.side_effect=ingest
        output.board.side_effect=lambda role,result,now:events.append(('board',result[0]['frame_id']))
        backend=MagicMock();backend.failed=False;backend.process.poll.return_value=None
        images=(np.zeros((8,8,3),np.uint8),)*2
        backend.receive.side_effect=[(dict(frame_id=n),images) for n in range(2)]
        worker=MagicMock();worker.future=None
        pool=MagicMock()
        def submit(fn,packet,images):
            future=Future();future.set_result((packet,images,[],None,10));return future
        pool.submit.side_effect=submit
        args=SimpleNamespace(seconds=0,frames=2,manual_record=False)
        with patch('ego2.multi_vio.app.zmq.Context'),patch('ego2.multi_vio.app.zmq.Poller'), \
             patch('ego2.multi_vio.app.Backend',return_value=backend), \
             patch('ego2.multi_vio.app.BoardWorker',return_value=worker), \
             patch('ego2.multi_vio.app.ThreadPoolExecutor') as factory,ExitStack() as resources:
            factory.return_value.__enter__.return_value=pool
            live({'mode':'multi_vio','timing':{'startup_timeout_s':30}},
                 {'chest':{'native':{}}},args,output,'unused',resources)
        self.assertEqual(events,[('vio',0),('vio',1),('board',0)])


if __name__=='__main__': unittest.main()
