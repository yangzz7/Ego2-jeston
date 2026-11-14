"""Six-axis relative camera attitude. No image, recorder, or viewer dependencies.

Hamilton wxyz quaternions map IMU vectors to an arbitrary local world with Z up.
Accelerometer specific force points UP at rest; do not subtract gravity first.
Heading is unobserved. Static initialization/bias adaptation require real rest.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def unit(q):
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError('Invalid quaternion norm')
    return np.asarray(q, dtype=float) / n


def rotation(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def from_rotation(r):
    r = np.asarray(r, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T@r, np.eye(3), atol=1e-4) or abs(np.linalg.det(r)-1) > 1e-4:
        raise ValueError('Camera/IMU extrinsic must be a proper rotation')
    # Largest-component construction is stable near rotations of 180 degrees.
    candidates = np.array([1+np.trace(r), 1+r[0,0]-r[1,1]-r[2,2],
                           1-r[0,0]+r[1,1]-r[2,2], 1-r[0,0]-r[1,1]+r[2,2]])
    index = int(np.argmax(candidates))
    if index == 0:
        q = [candidates[0], r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]]
    elif index == 1:
        q = [r[2,1]-r[1,2], candidates[1], r[0,1]+r[1,0], r[0,2]+r[2,0]]
    elif index == 2:
        q = [r[0,2]-r[2,0], r[0,1]+r[1,0], candidates[2], r[1,2]+r[2,1]]
    else:
        q = [r[1,0]-r[0,1], r[0,2]+r[2,0], r[1,2]+r[2,1], candidates[3]]
    return unit(q)


def increment(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return unit(np.r_[1., .5*np.asarray(vector)])
    return np.r_[math.cos(angle/2), np.asarray(vector)*(math.sin(angle/2)/angle)]


def align_up(acc):
    a = np.asarray(acc)/np.linalg.norm(acc)
    if a[2] < -1+1e-9:
        axis = np.cross(a, np.eye(3)[int(np.argmin(np.abs(a)))])
        return np.r_[0., axis/np.linalg.norm(axis)]
    return unit([1+a[2], a[1], -a[0], 0.])


@dataclass(frozen=True)
class Settings:
    kp: float = 1.0
    init_seconds: float = 2.5
    max_gap: float = .15
    sparse_gap: float = .015
    gravity: float = 9.80665
    static_gyro_limit: float = .015
    static_gyro_std: float = .006
    static_acc_std: float = .06
    bias_residual_limit: float = .0008
    bias_tau: float = 60.


class Mahony:
    """Kp feedback plus conservative rest-only bias adaptation; Ki is zero.

    One bounded window; expensive window checks at 10 Hz, not at IMU rate.
    Sustained linear acceleration/very slow yaw cannot be distinguished from
    gravity/bias in all cases with six axes. No absolute heading guarantee.
    """
    def __init__(self, camera_from_imu, settings=None):
        self.settings = settings or Settings()
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self.settings).values()):
            raise ValueError('Mahony settings must be finite and positive')
        if self.settings.sparse_gap >= self.settings.max_gap:
            raise ValueError('Sparse interval must be shorter than reset interval')
        self.q_ic = from_rotation(np.asarray(camera_from_imu).T)
        self.segment = 0
        self.valid_samples = 0
        self.resets = 0
        self.sparse_gaps = 0
        self.reset('WAITING: hold camera still for 2.5 s', count=False)

    def reset(self, reason='WAITING: data interrupted; hold still to rebuild reference', *, count=True):
        self.q = self.reference = self.previous = None
        self.bias = np.zeros(3)
        self.last_time = None
        self.last_gyro = None
        self.sparse_until = -math.inf
        self.recent_gap = 0.
        self.window = deque(maxlen=4096)
        self.last_check = -math.inf
        self.acc_weight = 0.
        self.status = reason
        if count:
            self.resets += 1

    def _stationary(self, now, acc, gyro):
        s = self.settings
        self.window.append((now, *acc, *gyro))
        while len(self.window) > 1 and self.window[1][0] <= now-s.init_seconds:
            self.window.popleft()
        if now-self.last_check < .1:
            return None
        self.last_check = now
        if len(self.window) < 100 or now-self.window[0][0] < s.init_seconds-.001:
            return None
        data = np.asarray(self.window)
        a, g = data[:,1:4], data[:,4:7]
        am, gm = a.mean(axis=0), g.mean(axis=0)
        limit = s.static_gyro_limit if self.q is None else s.bias_residual_limit
        # The long window alone lags motion onset. Gate the newest sample too.
        if self.q is not None and np.linalg.norm(gyro-self.bias) > 2*limit:
            return None
        if (abs(np.linalg.norm(am)-s.gravity) > .08*s.gravity or
                np.max(a.std(axis=0)) > s.static_acc_std or
                np.max(g.std(axis=0)) > s.static_gyro_std or
                np.linalg.norm(gm-self.bias) > limit):
            return None
        return am, gm

    def update(self, stamp, acc, gyro):
        stamp = float(stamp)
        a, g = np.asarray(acc, dtype=float), np.asarray(gyro, dtype=float)
        if a.shape != (3,) or g.shape != (3,) or not np.isfinite(np.r_[stamp, a, g]).all():
            self.reset('WAITING: invalid IMU; hold still to rebuild reference')
            return None
        if self.last_time is not None and stamp <= self.last_time:
            return None
        dt = 0. if self.last_time is None else stamp-self.last_time
        if dt > self.settings.max_gap + 1e-9:
            self.reset('WAITING: IMU gap; hold still to rebuild reference')
            dt = 0.
        sparse = dt > self.settings.sparse_gap + 1e-9
        if sparse:
            self.sparse_gaps += 1
            self.recent_gap = max(self.recent_gap if stamp < self.sparse_until else 0., dt)
            self.sparse_until = stamp + 1.
        self.last_time = stamp
        stationary = self._stationary(stamp, a, g)
        if self.q is None:
            if stationary is None:
                self.last_gyro = g.copy()
                return None
            am, self.bias = stationary
            self.q = align_up(am)
            self.reference = multiply(self.q, self.q_ic)
            self.previous = np.array([1., 0., 0., 0.])
            self.segment += 1
        else:
            # A bounded short gap keeps the original reference. Integrate the
            # mean of endpoint rates; no fictitious IMU samples are published.
            # Motion inside the gap is unknown, so label this as estimated.
            omega = ((self.last_gyro+g)*.5 if sparse and self.last_gyro is not None else g)-self.bias
            predicted = unit(multiply(self.q, increment(omega*dt)))
            norm = float(np.linalg.norm(a))
            # Compare current-time acceleration with current-time prediction,
            # not the preceding sample's attitude (which leads during turns).
            up = rotation(predicted)[2, :]  # R_WI.T @ world_up
            measured = a/norm if norm > 1e-9 else up
            angle = math.acos(float(np.clip(measured@up, -1., 1.)))
            # Norm and direction tests; smoothly vary feedback with trust.
            norm_weight = np.clip((.20-abs(norm/self.settings.gravity-1))/.15, 0., 1.) if norm > 1e-9 else 0.
            direction_weight = np.clip((math.radians(25)-angle)/math.radians(15), 0., 1.)
            target = float(norm_weight*direction_weight)
            tau = .02 if target < self.acc_weight else .3
            self.acc_weight += (1-math.exp(-dt/tau))*(target-self.acc_weight)
            error = np.cross(measured, up)
            # The current accelerometer is evidence only at the current time;
            # do not apply its correction repeatedly across a missing interval.
            feedback_dt = min(dt, self.settings.sparse_gap)
            self.q = unit(multiply(predicted,
                increment(self.settings.kp*self.acc_weight*error*feedback_dt)))
            if stationary is not None:
                # Window is checked every ~0.1 s. Never integrate gravity error
                # into bias; slow yaw remains fundamentally ambiguous.
                self.bias += (1-math.exp(-.1/self.settings.bias_tau))*(stationary[1]-self.bias)
        self.last_gyro = g.copy()
        quality = (f'ESTIMATED | IMU gaps <= {self.recent_gap*1000:.0f} ms'
                   if stamp < self.sparse_until else 'VALID | cam0 relative')
        self.status = quality + f' | reference #{self.segment}' + (' (rebuilt)' if self.segment > 1 else '')
        relative = unit(multiply(conjugate(self.reference), multiply(self.q, self.q_ic)))
        if relative@self.previous < 0:
            relative = -relative
        self.previous = relative
        self.valid_samples += 1
        return relative.copy()
