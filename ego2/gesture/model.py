"""Explicit, checksum-verified download of the official MediaPipe model v1."""
import hashlib
from pathlib import Path
import tempfile
import urllib.request


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
MODEL_SHA256 = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"


def download_model(destination):
    destination = Path(destination)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == MODEL_SHA256:
            return destination
        raise ValueError(f"Existing model differs from official v1; choose another --model path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as out:
            temporary = Path(out.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    out.write(block)
                    digest.update(block)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Model checksum mismatch; downloaded file was not installed")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
