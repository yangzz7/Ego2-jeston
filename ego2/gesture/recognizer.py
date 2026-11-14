"""Two independent MediaPipe VIDEO sessions, fed ordered paired camera frames."""
from dataclasses import dataclass, field
from pathlib import Path
import os
import sys

import numpy as np


def requested_backend(settings):
    value = os.environ.get("EGO2_GESTURE_BACKEND", settings.get("backend", "cpu"))
    if value not in ("cpu", "gpu", "auto"):
        raise ValueError("gesture.backend / EGO2_GESTURE_BACKEND must be cpu, gpu or auto")
    return value


@dataclass
class Detection:
    points: np.ndarray
    handedness: str = "Unknown"
    handedness_score: float = 0.0
    gesture: str = "Unknown"
    score: float = 0.0
    world_points: np.ndarray | None = None
    handedness_categories: list = field(default_factory=list)
    gesture_categories: list = field(default_factory=list)


class Recognizer:
    def __init__(self, settings):
        model = Path(settings["model"])
        if not model.is_file():
            raise FileNotFoundError(f"Missing model: {model}. Run python3 -m ego2.gesture --download-model")
        import mediapipe as mp
        self.mp = mp
        self.requested_backend = requested_backend(settings)
        self.backend = "cpu"
        self.fallbacks = 0
        self.task = None
        self._model = str(model)
        self._settings = dict(settings)
        if self.requested_backend in ("gpu", "auto"):
            try:
                self.task = self._create("gpu")
                self.backend = "gpu"
            except (RuntimeError, ValueError) as exc:
                if self.requested_backend == "gpu":
                    raise
                self._notice_fallback("initialization", exc)
        if self.task is None:
            self.task = self._create("cpu")
        print(f"[gesture-inference] {self.backend.upper()} (requested={self.requested_backend}); "
              "same model/input/thresholds; synchronous VIDEO", flush=True)

    def _create(self, backend):
        mp, settings = self.mp, self._settings
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._model,
                delegate=mp.tasks.BaseOptions.Delegate.GPU if backend == "gpu" else
                         mp.tasks.BaseOptions.Delegate.CPU),
            # Synchronous inference in the processing loop preserves stereo
            # pairing; VIDEO still enables MediaPipe's temporal hand tracker.
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=settings["num_hands"],
            min_hand_detection_confidence=settings["detection_confidence"],
            min_hand_presence_confidence=settings["presence_confidence"],
            min_tracking_confidence=settings["tracking_confidence"],
        )
        return mp.tasks.vision.GestureRecognizer.create_from_options(options)

    def _notice_fallback(self, stage, error):
        self.fallbacks += 1
        self.last_fallback = f'GPU {stage} failed: {type(error).__name__}: {str(error)[:1000]}'
        print(f"[gesture-inference] GPU {stage} failed; switching this eye to CPU: "
              f"{type(error).__name__}: {str(error)[:1000]}", file=sys.stderr, flush=True)

    def detect(self, rgb, timestamp_ms):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))
        try:
            result = self.task.recognize_for_video(image, timestamp_ms)
        except RuntimeError as exc:
            if self.backend != "gpu" or self.requested_backend != "auto":
                raise
            # An explicit GPU request fails visibly. In auto mode retry this
            # same image/time once on a new CPU VIDEO session. No stereo frame
            # is skipped and the external HandTracker keeps its own history.
            self._notice_fallback("inference (MediaPipe tracker resets)", exc)
            try:
                self.close()
            except RuntimeError as close_error:
                print(f"[gesture-inference] GPU cleanup: {close_error}", file=sys.stderr, flush=True)
            self.task = self._create("cpu")
            self.backend = "cpu"
            result = self.task.recognize_for_video(image, timestamp_ms)
        hands = []
        def categories(values):
            return [dict(index=int(c.index), category_name=c.category_name,
                         display_name=c.display_name, score=float(c.score)) for c in values]
        for i, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0] if result.handedness[i] else None
            gesture = result.gestures[i][0] if result.gestures[i] else None
            points = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            if points.shape != (21, 3) or not np.isfinite(points).all():
                continue
            name = gesture.category_name if gesture else "Unknown"
            hands.append(Detection(
                points=points,
                handedness=handedness.category_name if handedness else "Unknown",
                handedness_score=float(handedness.score) if handedness else 0.0,
                gesture="Unknown" if name in ("None", "Unknown", "") else name,
                score=float(gesture.score) if gesture else 0.0,
                world_points=np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]],
                                      dtype=np.float32) if i < len(result.hand_world_landmarks) else None,
                handedness_categories=categories(result.handedness[i]),
                gesture_categories=categories(result.gestures[i]),
            ))
        return hands

    def close(self):
        task, self.task = self.task, None
        if task is not None:
            task.close()
