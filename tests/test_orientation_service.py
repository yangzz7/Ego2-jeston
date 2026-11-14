import copy
import multiprocessing as mp
from pathlib import Path
import queue
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ego2.multi_vio.orientation_service import AttitudeInput,RoleProcessor,AttitudeService

G=9.80665
class ServiceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.cal=Path(self.tmp.name)/'cam.yaml'
  self.cal.write_text('cam0:\n  T_cam_imu: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n')
  self.profile={'calibration':{'file':str(self.cal)}}
 def batch(self,start=0,n=80):
  t=start+np.arange(n)*.005
  return dict(imu=np.column_stack([t,t,np.zeros(n),np.zeros(n),np.full(n,G),np.zeros((n,3))]),received=time.monotonic(),input_dropped=0)
 def test_capture_only_submits_owned_raw_measurements(self):
  event=mp.get_context('spawn').Event();event.set();q=queue.Queue()
  p=AttitudeInput(q,'head',100,event)
  pair=SimpleNamespace(received=101.,metadata=dict(image_timestamp_seconds=1.,imu_samples=[dict(timestamp_seconds=1.,acceleration_m_s2=[0,0,G],angular_velocity_rad_s=[0,0,0])]))
  original=copy.deepcopy(pair.metadata)
  with patch('ego2.common.mahony.Mahony.update',side_effect=AssertionError('fusion in capture')):p.capture(pair)
  data=q.get_nowait()['imu'];pair.metadata['imu_samples'][0]['acceleration_m_s2'][2]=0
  self.assertEqual(data.shape,(1,8));self.assertEqual(data[0,4],G);self.assertEqual(original['imu_samples'][0]['acceleration_m_s2'][2],G)
 def test_capture_backpressure_is_bounded_and_marks_break(self):
  event=mp.get_context('spawn').Event();event.set();q=queue.Queue(maxsize=1);q.put('full')
  p=AttitudeInput(q,'head',0,event)
  for i in range(100):
   t=1+i*.005
   p.capture(SimpleNamespace(received=t,metadata=dict(image_timestamp_seconds=t,imu_samples=[dict(timestamp_seconds=t,acceleration_m_s2=[0,0,G],angular_velocity_rad_s=[0,0,0])])))
  self.assertEqual(len(p.rows),64);self.assertEqual(p.stats['dropped'],36)
  q.get();p.flush();packet=q.get();self.assertTrue(np.isnan(packet['imu'][0,2:]).all());self.assertEqual(packet['input_dropped'],36)
 def test_service_stopping_does_not_touch_camera_or_recording(self):
  event=mp.get_context('spawn').Event();q=queue.Queue();p=AttitudeInput(q,'head',0,event)
  p.capture(None);self.assertTrue(p.disabled);self.assertTrue(q.empty())
 def test_all_ordered_samples_fused_and_long_backlog_breaks_reference(self):
  out=queue.Queue();p=RoleProcessor(out,'head',0,self.profile)
  for i in range(10):
   b=self.batch(i*.4);p.consume(b,b['received'])
  self.assertEqual(p.stats['samples'],800);self.assertGreater(p.stats['valid_quaternions'],100)
  self.assertEqual(p.publisher.estimator.segment,1)
  b=self.batch(4.);p.consume(b,b['received']+.5)
  self.assertEqual(p.stats['stale_batches'],1);self.assertIsNone(p.publisher.estimator.reference)
 def test_overflow_invalidates_before_next_samples(self):
  p=RoleProcessor(queue.Queue(),'head',0,self.profile)
  for i in range(10):
   b=self.batch(i*.4);p.consume(b,b['received'])
  b=self.batch(4.);b['input_dropped']=1;p.consume(b,b['received'])
  self.assertIsNone(p.publisher.estimator.reference);self.assertEqual(p.stats['input_dropped'],1)
 def test_worker_death_is_reported_and_close_does_not_hang(self):
  service=AttitudeService(mp.get_context('spawn'),{'head':self.profile},0,Path(self.tmp.name)/'service.log')
  try:
   self.assertTrue(service.ready.wait(10));service.process.terminate();service.process.join(timeout=5)
   self.assertFalse(service.healthy());service.close(lambda:None)
  finally:
   service.close(lambda:None)
   for q in service.outputs.values():q.close();q.join_thread()
 def test_four_camera_process_fairness_and_shutdown(self):
  roles=['head','chest','left_hand','right_hand'];got={r:[] for r in roles}
  service=AttitudeService(mp.get_context('spawn'),{r:self.profile for r in roles},0,Path(self.tmp.name)/'service.log')
  def drain():
   for r,q in service.outputs.items():
    try:
     while True:got[r].append(q.get_nowait())
    except queue.Empty:pass
  try:
   self.assertTrue(service.ready.wait(10))
   for i in range(10):
    for r in roles:service.inputs[r].put(self.batch(i*.4),timeout=10)
    # A nominal-rate source does not enqueue four seconds of IMU in one burst.
    # Bound each round here; overload/reset behavior has its own test above.
    deadline=time.monotonic()+5
    while min(x['samples'] for x in service.metrics().values())<(i+1)*80 and time.monotonic()<deadline:
     drain();time.sleep(.002)
    self.assertEqual(min(x['samples'] for x in service.metrics().values()),(i+1)*80)
   deadline=time.monotonic()+10
   while min(x['samples'] for x in service.metrics().values())<800 and time.monotonic()<deadline:drain();time.sleep(.01)
   service.close(drain)
   self.assertEqual(service.process.exitcode,0)
   for r,stats in service.metrics().items():
    self.assertEqual(stats['samples'],800,(r,stats));self.assertGreater(stats['valid_quaternions'],100)
    self.assertEqual(stats['reference_segments'],1);self.assertEqual(stats['input_dropped'],0)
    rows=np.concatenate([p['imu'] for p in got[r]])
    self.assertTrue((np.diff(rows[:,0])>0).all());self.assertTrue(np.isfinite(rows[-1,2:]).all())
  finally:
   service.close(drain)
   for q in service.outputs.values():q.close();q.join_thread()
if __name__=='__main__':unittest.main()
