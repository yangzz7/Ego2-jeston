# Third-party components

ORB-SLAM3 (including the local Cyperstereo adaptations) retains its upstream
GPLv3 license in `third_party/orb_slam3/LICENSE`. This standalone copy adds the
reconstruction snapshot/shutdown API and an atomic initialization-complete flag
in System.h, Map.h, Map.cc and LocalMapping.cc. Its SDK USB adapter is retained
under `third_party/orb_slam3/Thirdparty/usb`, with original copyright notices.

DBoW2, g2o and Sophus retain upstream notices and licenses in their respective
third-party folders. DBoW2 and g2o are built from these sources, not linked from
the original checkout. ORB vocabulary is downloaded to `vocabulary/ORBvoc.txt` using `tools/fetch_assets.py`; it is not committed to this source repository.

Google MediaPipe Gesture Recognizer float16 v1 model is downloaded and verified into
`models/gesture_recognizer.task`. Source:
https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

Model SHA-256: `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`.

Pangolin 0.6 source from the deployed Jetson is retained under `.native/src/Pangolin-0.6/`; its license is `.native/src/Pangolin-0.6/LICENCE`. Compiled libraries are excluded.
