"""Attitude correctness against independent Rodrigues rotations, failure isolation."""
import copy
import io
import math
from pathlib import Path
import queue
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import rerun as rr

from ego2.common.mahony import Mahony, Settings, rotation, from_rotation, increment
from ego2.multi_vio.orientation_preview import QuaternionPublisher, QuaternionLogger

G = 9.80665
DT = .005


def rodrigues(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v/angle
    k = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)


def initialize(est, initial=np.eye(3), bias=np.zeros(3), start=0.):
    q = None
    for t in np.arange(start, start+3.+DT/2, DT):
        q = est.update(t, initial.T@np.array([0.,0.,G]), bias)
    assert q is not None, est.status
    return float(t)


class FusionTests(unittest.TestCase):
    def test_all_static_orientations_initialize_relative_identity(self):
        for v in ([0,0,0],[math.pi,0,0],[0,math.pi,0],[0,0,math.pi],[1.2,-.6,2.]):
            est = Mahony(rodrigues(np.array([.8,.1,-.2])))
            initialize(est, rodrigues(np.array(v)), np.array([.002,-.003,.001]))
            np.testing.assert_allclose(est.previous, [1,0,0,0], atol=1e-12)
            np.testing.assert_allclose(rotation(est.q).T@[0,0,G], rodrigues(np.array(v)).T@[0,0,G], atol=1e-8)

    def test_rotations_and_nontrivial_camera_extrinsic_direction(self):
        rci = rodrigues(np.array([1.4,-.3,.2]))
        for axis in np.eye(3):
            est = Mahony(rci)
            bias = np.array([.002,-.003,.001])
            t = initialize(est, bias=bias)
            for n in range(1,501):
                r = rodrigues(axis*.6*n*DT)
                q = est.update(t+n*DT, r.T@[0,0,G], axis*.6+bias)
            expected = rci@r@rci.T
            np.testing.assert_allclose(rotation(q), expected, atol=.004)
            self.assertAlmostEqual(np.linalg.norm(q), 1., places=12)

    def test_multiple_turns_sign_continuity(self):
        est = Mahony(np.eye(3)); t = initialize(est)
        previous = est.previous
        for n in range(1,2401):
            q = est.update(t+n*DT, [0,0,G], [0,0,1.])
            self.assertGreater(q@previous, .999)
            previous = q
        np.testing.assert_allclose(rotation(q), rodrigues(np.array([0,0,12.])), atol=1e-10)

    def test_slow_yaw_is_not_learned_as_bias(self):
        est = Mahony(np.eye(3)); t = initialize(est)
        for n in range(1,1001):
            q = est.update(t+n*DT, [0,0,G], [0,0,.004])
        np.testing.assert_allclose(rotation(q), rodrigues(np.array([0,0,.02])), atol=1e-9)
        np.testing.assert_array_equal(est.bias, 0.)

    def test_gravity_corrects_tilt_even_with_zero_gyro(self):
        est = Mahony(np.eye(3)); t = initialize(est)
        est.q = increment(np.array([.08,0,0]))
        for n in range(1,1601): est.update(t+n*DT, [0,0,G], [0,0,0])
        self.assertLess(np.linalg.norm(rotation(est.q)-np.eye(3)), .001)

    def test_linear_acceleration_rejected_and_recovers(self):
        est = Mahony(np.eye(3)); t = initialize(est)
        for n in range(1,101): est.update(t+n*DT, [0,0,G], [0,0,0])
        for n in range(101,301): est.update(t+n*DT, [G,0,G], [0,0,0])
        self.assertLess(est.acc_weight, .001)
        self.assertLess(np.linalg.norm(rotation(est.q)-np.eye(3)), .03)
        for n in range(301,2301): est.update(t+n*DT, [0,0,G], [0,0,0])
        self.assertGreater(est.acc_weight, .99)
        self.assertLess(np.linalg.norm(rotation(est.q)-np.eye(3)), .001)

    def test_motion_blocks_initialization(self):
        est = Mahony(np.eye(3))
        for t in np.arange(0,4,DT): self.assertIsNone(est.update(t,[0,0,G],[0,0,.03]))
        self.assertEqual(est.segment, 0)

    def test_stationary_four_camera_noise_and_sparse_intervals_initialize(self):
        # Observed chest gyro noise is ~0.0047 rad/s, above the old 0.004
        # threshold. Alternating zero-mean noise is independent of the filter.
        est = Mahony(np.eye(3)); stamp=0.; outputs=[]
        for n in range(650):
            stamp += .075 if n % 70 == 69 else .005
            q = est.update(stamp, [0,0,G], [.005*(-1)**n,.004,0])
            if q is not None: outputs.append(q)
        self.assertGreater(len(outputs),100)
        self.assertEqual(est.segment,1)
        self.assertEqual(est.resets,0)
        self.assertIn('ESTIMATED',est.status)
        self.assertGreater(est.sparse_gaps,0)

    def test_sparse_turn_preserves_reference_and_flags_uncertain_motion(self):
        est=Mahony(np.eye(3)); t=initialize(est)
        # Establish the constant angular rate before the first missing block.
        elapsed=0.; previous=est.previous
        for dt in [.005]*10+[.005,.075,.005,.04]*60:
            elapsed+=dt
            q=est.update(t+elapsed,[0,0,G],[0,0,.4])
            self.assertIsNotNone(q)
            self.assertGreater(q@previous,.999)
            previous=q
        np.testing.assert_allclose(rotation(q),rodrigues(np.array([0,0,.4*elapsed])),atol=1e-10)
        self.assertEqual(est.segment,1)
        self.assertEqual(est.resets,0)
        self.assertIn('ESTIMATED',est.status)
        for _ in range(220):
            elapsed+=DT;est.update(t+elapsed,[0,0,G],[0,0,.4])
        self.assertIn('VALID',est.status)

    def test_noisy_motion_still_blocks_initialization(self):
        est=Mahony(np.eye(3))
        for n in range(800):
            self.assertIsNone(est.update(n*DT,[0,0,G],[.012*(-1)**n,0,0]))
        self.assertEqual(est.segment,0)

    def test_sparse_three_axis_turn_uses_current_time_gravity_prediction(self):
        for axis in np.eye(3):
            est=Mahony(np.eye(3)); t=initialize(est);elapsed=0.
            for dt in [.005]*10+[.005,.075,.005,.04]*60:
                elapsed+=dt
                expected=rodrigues(axis*.4*elapsed)
                q=est.update(t+elapsed,expected.T@[0,0,G],axis*.4)
            np.testing.assert_allclose(rotation(q),expected,atol=1e-10)
            self.assertEqual(est.segment,1)
            self.assertEqual(est.resets,0)

    def test_device_dt_not_nominal_rate(self):
        est = Mahony(np.eye(3)); t = initialize(est)
        elapsed=0.
        for dt in [.003,.007,.004,.006]*150:
            elapsed+=dt
            q = est.update(t+elapsed,[0,0,G],[0,0,.2])
        np.testing.assert_allclose(rotation(q), rodrigues(np.array([0,0,.2*elapsed])), atol=1e-10)

    def test_gap_invalidates_and_rebuilds_explicit_reference(self):
        est = Mahony(np.eye(3)); t = initialize(est)
        self.assertIsNone(est.update(t+.2,[0,0,G],[0,0,0]))
        self.assertIn('gap',est.status)
        initialize(est,start=t+.2)
        self.assertEqual(est.segment,2)
        self.assertIn('rebuilt',est.status)
        np.testing.assert_allclose(est.previous,[1,0,0,0],atol=1e-12)

    def test_duplicate_and_invalid_samples(self):
        est = Mahony(np.eye(3)); t = initialize(est)
        saved=est.previous.copy()
        self.assertIsNone(est.update(t,[0,0,G],[0,0,1]))
        np.testing.assert_array_equal(est.previous,saved)
        self.assertIsNone(est.update(t+DT,[0,0,float('nan')],[0,0,0]))
        self.assertIsNone(est.reference)

    def test_bad_extrinsics_and_settings_rejected(self):
        for r in [np.diag([-1,1,1]),np.eye(4),np.zeros((3,3))]:
            with self.assertRaises(ValueError): Mahony(r)
        with self.assertRaises(ValueError): Mahony(np.eye(3),Settings(kp=float('nan')))

    def test_rotation_conversion_near_pi(self):
        for v in ([math.pi,0,0],[0,math.pi,0],[0,0,math.pi],[1.81,1.81,1.81]):
            r=rodrigues(np.array(v)); np.testing.assert_allclose(rotation(from_rotation(r)),r,atol=1e-12)


def make_pair(t, seq, *, gap=False):
    samples=[dict(timestamp_seconds=t+i*DT,acceleration_m_s2=[0,0,G],angular_velocity_rad_s=[0,0,.002],accepted=True) for i in range(4)]
    return SimpleNamespace(received=100+t+.1,metadata=dict(image_timestamp_seconds=t+3*DT,imu_samples=[] if gap else samples),sequence=seq)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.cal=Path(self.tmp.name)/'camera.yaml'
        self.cal.write_text('cam0:\n  T_cam_imu: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n')
        self.profile=dict(calibration=dict(file=str(self.cal)))

    def test_fuses_all_samples_before_backpressure_and_preserves_input(self):
        pending=queue.Queue(maxsize=1); pending.put('busy')
        pub=QuaternionPublisher(pending,'chest',100.,self.profile,capacity=12)
        with redirect_stdout(io.StringIO()):
            for i in range(180):
                p=make_pair(i*.02,i); before=copy.deepcopy(p.metadata);pub.capture(p)
                self.assertEqual(p.metadata,before)
        self.assertEqual(pub.stats['samples'],720)
        self.assertGreater(pub.stats['valid_quaternions'],100)
        self.assertEqual(pub.stats['reference_segments'],1)
        self.assertGreater(pub.stats['dropped'],0)
        pending.get();pub.flush();packet=pending.get()
        self.assertEqual(packet['imu'].shape,(12,6))
        self.assertTrue(np.isnan(packet['imu'][0,2:]).all())
        np.testing.assert_allclose(packet['imu'][-1,2:],[1,0,0,0],atol=1e-12)

    def test_missing_calibration_reports_error_without_stopping_capture(self):
        pub=QuaternionPublisher(queue.Queue(),'chest',100.,{})
        with redirect_stdout(io.StringIO()): pub.capture(make_pair(0,1))
        p=pub.queue.get()
        self.assertTrue(np.isnan(p['imu'][:,2:]).all())
        self.assertIn('CALIBRATION ERROR',p['status'][0][1])
        self.assertFalse(pub.disabled)

    def test_missing_metadata_and_clock_reset_reinitialize(self):
        pub=QuaternionPublisher(queue.Queue(),'chest',100.,self.profile)
        with redirect_stdout(io.StringIO()):
            for i in range(160): pub.capture(make_pair(i*.02,i))
            self.assertEqual(pub.estimator.segment,1)
            pub.capture(make_pair(3.2,160,gap=True))
            self.assertIsNone(pub.estimator.reference)
            p=make_pair(.1,161); p.received=104.;pub.capture(p)
        self.assertEqual(pub.stats['resets'],1)
        self.assertIsNone(pub.estimator.reference)

    def test_serialization_status_and_explicit_times(self):
        pub=QuaternionPublisher(queue.Queue(),'chest',100.,self.profile)
        with redirect_stdout(io.StringIO()):
            for i in range(160):pub.capture(make_pair(i*.02,i))
        pub.flush();packets=[]
        while not pub.queue.empty(): packets.append(pub.queue.get())
        mock=MagicMock();logger=QuaternionLogger(mock,['chest'])
        for p in packets:logger.consume(p)
        mock.set_time.assert_not_called();mock.reset_time.assert_not_called()
        transitions=[s for p in packets for _,s in p['status']]
        self.assertTrue(any('WAITING' in s for s in transitions))
        self.assertTrue(any('VALID' in s for s in transitions))
        rec=rr.RecordingStream('quaternion-test');path=Path(self.tmp.name)/'data.rrd';rec.save(path)
        logger=QuaternionLogger(rec,['chest'])
        for p in packets:logger.consume(p)
        rec.flush();rec.disconnect();self.assertGreater(path.stat().st_size,0)
        with self.assertRaises(ValueError):logger.consume(dict(packets[0],role='other'))


if __name__=='__main__':unittest.main()
