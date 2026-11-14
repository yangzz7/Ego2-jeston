"""Delegate policy, paired VIDEO state and failure behavior without a camera."""
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from ego2.gesture.recognizer import Recognizer, requested_backend


class Task:
    def __init__(self, failure=None):
        self.failure, self.calls, self.closed = failure, [], 0

    def recognize_for_video(self, image, stamp):
        self.calls.append((image, stamp))
        if self.failure:
            raise self.failure
        return NS(hand_landmarks=[], handedness=[], gestures=[], hand_world_landmarks=[])

    def close(self):
        self.closed += 1


class BaseOptions:
    Delegate = NS(CPU=0, GPU=1)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Backends(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        model = Path(self.tmp.name)/'model.task'
        model.write_bytes(b'model')
        self.settings = dict(model=str(model), num_hands=2, detection_confidence=.5,
                             presence_confidence=.5, tracking_confidence=.5)
        self.options = []
        self.tasks = []
        self.create_error = None
        self.gpu_error = None
        self.cpu_error = None
        def create(options):
            self.options.append(options)
            gpu = options.base_options.delegate == 1
            if gpu and self.create_error: raise self.create_error
            task = Task(self.gpu_error if gpu else self.cpu_error)
            self.tasks.append(task)
            return task
        module = NS(tasks=NS(BaseOptions=BaseOptions, vision=NS(
            GestureRecognizerOptions=lambda **kw:NS(**kw),
            RunningMode=NS(VIDEO='VIDEO'),
            GestureRecognizer=NS(create_from_options=create))),
            Image=lambda **kw:NS(**kw), ImageFormat=NS(SRGB='SRGB'))
        for context in (patch.dict('sys.modules', mediapipe=module),
                        patch.dict('os.environ', {}, clear=True)):
            context.start()
            self.addCleanup(context.stop)

    def make(self, backend='auto'):
        r = Recognizer({**self.settings,'backend':backend})
        self.addCleanup(r.close)
        return r

    def test_gpu_uses_same_model_parameters_and_independent_video_sessions(self):
        a,b = self.make(),self.make()
        self.assertIsNot(a.task,b.task)
        self.assertEqual(a.backend,'gpu')
        for opt in self.options:
            self.assertEqual(opt.base_options.model_asset_path,self.settings['model'])
            self.assertEqual(opt.running_mode,'VIDEO')
            self.assertEqual(opt.num_hands,2)
            self.assertEqual(opt.min_hand_detection_confidence,.5)
            self.assertEqual(opt.min_hand_presence_confidence,.5)
            self.assertEqual(opt.min_tracking_confidence,.5)
        raw = np.arange(6*8*3,dtype=np.uint8).reshape(6,8,3)
        saved=raw.copy()
        for stamp in (10,20,40):
            a.detect(raw[:,::2],stamp)
            b.detect(raw,stamp)
        np.testing.assert_array_equal(raw,saved)
        self.assertEqual([t for _,t in a.task.calls],[10,20,40])

    def test_cpu_override_never_creates_gpu(self):
        with patch.dict('os.environ',EGO2_GESTURE_BACKEND='cpu'):
            self.assertEqual(self.make().backend,'cpu')
        self.assertEqual([o.base_options.delegate for o in self.options],[0])

    def test_auto_creation_failure_falls_back(self):
        self.create_error=RuntimeError('EGL unavailable')
        r=self.make()
        self.assertEqual((r.backend,r.fallbacks),('cpu',1))
        self.assertEqual([o.base_options.delegate for o in self.options],[1,0])

    def test_explicit_gpu_creation_failure_is_not_hidden(self):
        self.create_error=RuntimeError('EGL unavailable')
        with self.assertRaisesRegex(RuntimeError,'EGL'):self.make('gpu')
        self.assertEqual(len(self.options),1)

    def test_auto_runtime_retries_identical_frame_once_then_stays_cpu(self):
        self.gpu_error=RuntimeError('GPU lost')
        r=self.make();old=r.task
        rgb=np.zeros((6,8,3),np.uint8)
        self.assertEqual(r.detect(rgb,42),[])
        self.assertEqual((r.backend,r.fallbacks,old.closed),('cpu',1,1))
        self.assertIs(old.calls[0][0],r.task.calls[0][0])
        self.assertEqual(r.task.calls[0][1],42)
        r.detect(rgb,43)
        self.assertEqual(len(self.options),2)
        cpu=r.task;r.close();r.close()
        self.assertEqual(cpu.closed,1)

    def test_invalid_timestamp_is_not_hidden_by_gpu_fallback(self):
        self.gpu_error=ValueError('timestamp must increase')
        r=self.make()
        with self.assertRaises(ValueError):r.detect(np.zeros((2,2,3),np.uint8),0)
        self.assertEqual(r.fallbacks,0)

    def test_cpu_error_after_fallback_propagates(self):
        self.gpu_error=RuntimeError('GPU lost');self.cpu_error=RuntimeError('CPU failed')
        r=self.make()
        with self.assertRaisesRegex(RuntimeError,'CPU failed'):
            r.detect(np.zeros((2,2,3),np.uint8),0)
        self.assertEqual(len(self.options),2)

    def test_explicit_gpu_runtime_failure_is_not_hidden(self):
        self.gpu_error=RuntimeError('GPU lost');r=self.make('gpu')
        with self.assertRaisesRegex(RuntimeError,'GPU lost'):
            r.detect(np.zeros((2,2,3),np.uint8),0)
        self.assertEqual(r.fallbacks,0)

    def test_policy_validation_and_old_profile_cpu_default(self):
        self.assertEqual(requested_backend({}),'cpu')
        with self.assertRaises(ValueError):requested_backend({'backend':'cuda'})
        with patch.dict('os.environ',EGO2_GESTURE_BACKEND=''):
            with self.assertRaises(ValueError):requested_backend({})


if __name__=='__main__':unittest.main()
