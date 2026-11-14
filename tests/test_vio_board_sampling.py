"""Sparse stereo board observations must accumulate without weakening quality."""
from collections import deque
import copy
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
import numpy as np

from ego2.multi_vio.app import Output
from ego2.multi_vio.alignment import Coordinator
from tests import test_vio_board_latency as latency


class SamplingTests(unittest.TestCase):
    def helper(self, **alignment):
        t=latency.BoardLatencyTests();t.setUp();t.cfg['alignment'].update(alignment)
        t.c=Coordinator(t.cfg,t.devices);t.state=t.c.devices['chest']
        return t

    def simulate(self,t,*,move_board=False,poor_stereo=False):
        pending=deque();peak=0
        # VIO at 5 Hz, board attempts at 2.5 Hz, two rejected attempts per
        # five. Each measurement remains tied to its own captured pose.
        for n in range(201):
            packet,board=t.ingest(n,stamp=n*.2)
            now=n*.2+.1
            if n%2==0:
                if move_board:board.pose[0,3]+=.1*(n//2)
                if poor_stereo or (n//2)%5==3:
                    board.inliers=frozenset(k for k in board.inliers if k[0]=='cam1')
                if (n//2)%5==0:board=None
                pending.append((n*.2+.65,(packet,board)))
            while pending and pending[0][0]<=now:
                _,pair=pending.popleft();t.observe(pair,now)
                peak=max(peak,len(t.state.window.samples))
            t.c.relative(now)
            if t.c.started:return now,peak
        return None,peak

    def test_sparse_real_pairs_reach_twenty_without_changing_transform(self):
        t=self.helper()
        when,peak=self.simulate(t)
        self.assertIsNotNone(when)
        self.assertGreater(when,8.)
        self.assertGreaterEqual(peak,20)
        np.testing.assert_allclose(t.state.anchor,t.anchor,atol=1e-9)
        self.assertEqual(t.cfg['alignment']['min_tags_per_eye'],4)
        self.assertEqual(t.cfg['timing']['stale_s'],.6)

    def test_previous_window_reproduces_never_finishing(self):
        t=self.helper(max_duration_s=8.,max_sample_gap_s=.6)
        when,peak=self.simulate(t)
        self.assertIsNone(when);self.assertLess(peak,20)
        self.assertEqual(t.state.window.last_reset,'board_sample_gap')

    def test_eight_second_duration_is_insufficient_even_without_gap_resets(self):
        t=self.helper(max_duration_s=8.,max_sample_gap_s=2.)
        when,peak=self.simulate(t)
        self.assertIsNone(when);self.assertLess(peak,20)
        self.assertGreater(t.state.window.expired_samples,0)

    def test_longer_window_still_rejects_one_eye_support(self):
        t=self.helper();when,peak=self.simulate(t,poor_stereo=True)
        self.assertIsNone(when);self.assertEqual(peak,0)

    def test_moving_board_cannot_pass_existing_consensus_thresholds(self):
        t=self.helper();when,peak=self.simulate(t,move_board=True)
        self.assertIsNone(when);self.assertGreaterEqual(peak,20)
        self.assertIsNone(t.state.anchor)
        self.assertEqual(t.state.window.quality['reason'],'unstable_alignment')

    def test_gap_above_new_limit_still_discards_old_samples(self):
        t=self.helper();t.observe(t.ingest(0),.1)
        for n in range(1,24):last=t.ingest(n)
        t.observe(last,2.4)
        self.assertEqual(len(t.state.window.samples),1)
        self.assertEqual(t.state.window.last_reset,'board_sample_gap')
        self.assertEqual(t.state.window.last_reset_samples,1)

    def test_real_tracking_loss_still_clears_window_immediately(self):
        t=self.helper();t.observe(t.ingest(0),.1);t.ingest(1,valid=False)
        self.assertFalse(t.state.window.samples)
        self.assertEqual(t.state.window.last_reset,'tracking_invalid')

    def test_diagnostics_distinguish_observed_and_inlier_tags_from_overlays(self):
        t=self.helper();packet,board=t.ingest(0)
        args=SimpleNamespace(demo=False)
        output=Output(t.cfg,t.devices,args,MagicMock(),None)
        output.ingest('chest',packet,.1)
        observations=[SimpleNamespace(camera=cam,marker_id=i,reason='ok',source=source)
                      for cam in ('cam0','cam1')
                      for i,source in [(0,'decoded'),(5,'flow'),(30,'decoded'),(35,'flow'),(2,'board'),(3,'board')]]
        observations.append(SimpleNamespace(camera='cam0',marker_id=7,reason='ok',source='decoded'))
        original=copy.deepcopy(board.inliers)
        output.board('chest',(packet,None,observations,SimpleNamespace(measurement=board),300),.2)
        self.assertEqual(output.board_support['chest'],{'cam0':(5,4),'cam1':(4,4)})
        self.assertEqual(board.inliers,original)
        text=io.StringIO()
        with redirect_stdout(text):output.progress(.2)
        self.assertIn('cam0=4/5 cam1=4/4',text.getvalue())
        self.assertIn('last_reset=none(0)',text.getvalue())


if __name__=='__main__':unittest.main()
