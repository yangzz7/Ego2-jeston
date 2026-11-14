import math
import unittest
import numpy as np
from ego2.common.mahony import Mahony,rotation
from ego2.common.mahony_native import NativeMahony,status_text
from tests.test_mahony import rodrigues,G

class NativeTests(unittest.TestCase):
 def compare(self,rows,r=np.eye(3),batch=7):
  python=Mahony(r);native=NativeMahony(r);self.addCleanup(native.close)
  for begin in range(0,len(rows),batch):
   part=rows[begin:begin+batch];q,infos=native.update_batch(part)
   for row,actual,info in zip(part,q,infos):
    if np.isfinite(row[1:]).all():expected=python.update(row[1],row[2:5],row[5:8])
    else:python.reset();expected=None
    self.assertEqual(status_text(info),python.status)
    self.assertEqual(tuple(map(int,info[[1,2,3,5]])),(python.segment,python.resets,python.sparse_gaps,python.valid_samples))
    if expected is None:self.assertTrue(np.isnan(actual).all())
    else:
     np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
     self.assertAlmostEqual(np.linalg.norm(actual),1.,places=12)
 def series(self):
  rng=np.random.default_rng(19);rows=[];t=0.;R=rodrigues(np.array([.8,-.3,.2]))
  for k in range(3000):
   dt=.075 if k>600 and k%37==0 else .005
   t+=dt
   if k in (1700,2100):t+=.3
   gyro=np.zeros(3) if k<650 or k>=1700 else np.array([.2*math.sin(k*.003),.25,.35])
   R=R@rodrigues(gyro*dt)
   acc=R.T@[0,0,G]
   if 1400<k<1550:acc=acc+[2,0,5]
   rows.append([t,t,*acc,*(gyro+np.array([.002,-.001,.003])+rng.normal(0,.001,3))])
  rows=np.array(rows)
  rows[2000,2]=math.nan
  rows[1200,1]=rows[1199,1]
  return rows
 def test_batched_matches_reference_noise_motion_gaps_acceleration(self):
  self.compare(self.series(),rodrigues(np.array([1.2,.3,-.1])),7)
 def test_batch_boundaries_and_four_independent_rotations(self):
  rows=self.series()
  for r in [np.eye(3),rodrigues(np.array([math.pi,0,0])),rodrigues(np.array([0,math.pi,0]))]:self.compare(rows[:1000],r,80)
 def test_long_yaw_sign_continuity(self):
  t=np.arange(3600)*.005;g=np.zeros((3600,3));g[650:,2]=2.
  self.compare(np.column_stack([t,t,np.zeros((3600,2)),np.full(3600,G),g]),batch=13)
 def test_invalid_shape_and_bad_rotation_rejected(self):
  with self.assertRaises(ValueError):NativeMahony(np.zeros((3,3)))
  n=NativeMahony(np.eye(3));self.addCleanup(n.close)
  with self.assertRaises(ValueError):n.update_batch([[1,2]])
if __name__=='__main__':unittest.main()
