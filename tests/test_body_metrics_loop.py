"""Run the supervisor past its periodic diagnostics with the real body summary."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from ego2.multi_vio import multi_mode
from ego2.multi_vio.body_preview import BodyPreview, load_settings


class MetricsLoopTests(unittest.TestCase):
    def test_one_and_four_camera_supervisor_reports_real_body_summary_and_exits(self):
        for roles in (('chest',),('chest','head','left_hand','right_hand')):
            with self.subTest(roles=roles):
                snapshots=[];clock=[-6.]
                def monotonic():clock[0]+=6.;return clock[0]
                class Pending(queue.Queue):
                    def close(self):pass
                    def join_thread(self):pass
                class Process:
                    exitcode=0
                    def __init__(self,*,target,name,args):self.root=Path(args[4])
                    def start(self):
                        (self.root/'result.json').write_text(json.dumps(dict(status='finished')))
                    def is_alive(self):return clock[0]<20.
                    def join(self,timeout=None):
                        metrics=self.root.parent/'body-preview-metrics.json'
                        if metrics.exists():snapshots.append(json.loads(metrics.read_text()))
                context=SimpleNamespace(Event=threading.Event,Queue=Pending,Process=Process)
                body=BodyPreview(MagicMock(),roles,settings=load_settings())
                viewer=SimpleNamespace(body_preview=body,display_processors={},raw_imu_logger=None,
                                       tick=body.tick,state=MagicMock(),close=MagicMock())
                args=SimpleNamespace(mode='gesture',demo=True,record=False,no_viz=False,headless=False)
                devices={r:dict(native=dict(serial='synthetic-'+r),profile={}) for r in roles}
                output=io.StringIO()
                with redirect_stdout(output), \
                     patch.object(multi_mode,'UnifiedViewer',return_value=viewer), \
                     patch.object(multi_mode.mp,'get_context',return_value=context), \
                     patch.object(multi_mode,'time',SimpleNamespace(monotonic=monotonic,sleep=lambda _:None)):
                    result=multi_mode.run(dict(recording=dict(max_gib=1)),devices,args)
                self.assertTrue(snapshots,'Periodic diagnostics branch was not reached')
                self.assertEqual(set(snapshots[0]['states']),set(roles))
                self.assertIn('models',snapshots[0])
                self.assertIn('selector',snapshots[0])
                self.assertIn('rendered',snapshots[0])
                self.assertIn('[body-preview] drawn=',output.getvalue())
                self.assertIsNone(result['error'])
                self.assertEqual(set(result['results']),set(roles))
                viewer.close.assert_called_once()

if __name__=='__main__':unittest.main()
