"""Native batched arithmetic behind the unchanged quaternion plotting protocol."""
import os
import numpy as np
from .orientation_preview import QuaternionPublisher
from .raw_imu import RawImuPublisher

class NativeQuaternionPublisher(QuaternionPublisher):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        backend=os.environ.get('EGO2_ATTITUDE_BACKEND','auto')
        if backend not in ('auto','python','native'):raise ValueError('Invalid EGO2_ATTITUDE_BACKEND')
        self.native=False
        if self.estimator is not None and backend!='python':
            try:
                from ..common.mahony_native import NativeMahony
                self.estimator=NativeMahony(np.asarray(self.meta['camera_from_imu_rotation']),self.estimator.settings)
                self.native=True
            except (OSError,RuntimeError) as exc:
                if backend=='native':raise
                self.meta['backend_fallback']=str(exc)
                print(f'[attitude] {self.role}: native unavailable, using Python: {exc}',flush=True)
        self.meta['fusion_backend']='native-cpu' if self.native else 'python'
        print(f'[attitude] {self.role}: fusion={self.meta["fusion_backend"]}; independent process',flush=True)

    def append_rows(self,rows):
        if not self.native:
            for row in rows:self._append(row)
            return
        from ..common.mahony_native import status_text
        quaternions,info=self.estimator.update_batch(rows)
        # Read the existing native reference once per segment, never refit it
        # from a moving accelerometer. If a batch spans a reset, only its last
        # segment uses this reference; earlier chart rows remain unchanged.
        if len(info) and np.isfinite(quaternions[-1]).all():
            valid = np.isfinite(quaternions).all(axis=1) & (info[:,1] == info[-1,1])
            self._body_reference(float(rows[np.flatnonzero(valid)[0]][0]), info[-1,1])
        elif len(info):
            self.body_reference = None
        for row,q,entry in zip(rows,quaternions,info):
            status=status_text(entry)
            if status!=self.last_status:
                print(f'[attitude] {self.role}: {status}',flush=True)
                self.events.append((float(row[0]),status));self.events=self.events[-100:]
                self.last_status=status
            RawImuPublisher._append(self,(float(row[0]),float(row[1]),*q))
        self.stats.update(valid_quaternions=self.estimator.valid_samples,
            reference_segments=self.estimator.segment,attitude_resets=self.estimator.resets,
            sparse_imu_gaps=self.estimator.sparse_gaps)
