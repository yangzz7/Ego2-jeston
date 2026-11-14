"""Per-eye perspective inference with landmarks mapped to unmodified input pixels."""
import cv2
import numpy as np

from ..common.config import orient
from ..common.geometry import Maps, View, project, rotation


def undo_orientation(points, raw_size, channel):
    """Inverse of config.orient for floating-point pixel coordinates."""
    width, height = raw_size
    turns = channel.get("rotate_cw", 0) // 90
    ow, oh = (height, width) if turns % 2 else (width, height)
    x, y = np.asarray(points, dtype=np.float64).T.copy()
    if channel.get("flip_horizontal", False):
        x = ow - 1 - x
    if channel.get("flip_vertical", False):
        y = oh - 1 - y
    if turns == 1:
        x, y = y, height - 1 - x
    elif turns == 2:
        x, y = width - 1 - x, height - 1 - y
    elif turns == 3:
        x, y = width - 1 - y, x
    return np.column_stack((x, y)).astype(np.float32)


class Rectifier:
    def __init__(self, camera, channel, settings):
        self.channel = channel
        self.view = View(**settings, distance=0)
        self.camera = camera
        self.mask = np.full((camera.size[1], camera.size[0]), 255, np.uint8)
        if channel.get("valid_mask"):
            mask = cv2.imread(channel["valid_mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != self.mask.shape:
                raise ValueError("valid_mask must have calibration resolution")
            self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        rows = int(channel.get("invalid_bottom_rows", 0))
        if not 0 <= rows < camera.size[1]:
            raise ValueError("invalid_bottom_rows outside image")
        if rows:
            self.mask[-rows:] = 0
        self.maps = None
        self.raw_size = None
        from .preprocess import Accelerator
        self.accelerator = Accelerator("rectification RGB")

    def _build(self, size):
        view, cam, ch = self.view, self.camera, self.channel
        yy, xx = np.mgrid[:view.height, :view.width].astype(np.float64)
        f = view.width / (2 * np.tan(np.deg2rad(view.hfov / 2)))
        rays = np.stack(((xx - (view.width - 1) / 2) / f,
                         (yy - (view.height - 1) / 2) / f, np.ones_like(xx)), axis=-1)
        u, v, valid = project(cam, rays @ rotation(view.yaw, view.pitch).T)
        valid &= cv2.remap(self.mask, u.astype(np.float32), v.astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) == 255
        x, y, cw, chh = map(float, ch.get("calibration_crop", [0, 0, *cam.size]))
        if not np.isfinite([x, y, cw, chh]).all() or min(cw, chh) <= 0 or min(x, y) < 0 or x + cw > cam.size[0] or y + chh > cam.size[1]:
            raise ValueError("Invalid calibration_crop")
        iw, ih = size
        if not ch.get("allow_anisotropic_resize", False) and abs((iw / ih) / (cw / chh) - 1) > 0.002:
            raise ValueError("Input aspect ratio differs from calibration; declare calibration_crop")
        mx = ((u - x + 0.5) * iw / cw - 0.5).astype(np.float32)
        my = ((v - y + 0.5) * ih / chh - 0.5).astype(np.float32)
        valid &= (mx >= 0) & (mx < iw - 1) & (my >= 0) & (my < ih - 1)
        mx[~valid], my[~valid] = -1, -1
        return Maps(view, [(mx, my)], [valid], (size,))

    def prepare(self, raw_bgr):
        self.raw_size = (raw_bgr.shape[1], raw_bgr.shape[0])
        calibrated = orient(raw_bgr, self.channel)
        size = (calibrated.shape[1], calibrated.shape[0])
        if self.maps is None or self.maps.sizes != (size,):
            self.accelerator.close()
            self.maps = self._build(size)
            if np.mean(self.maps.valid[0]) < 0.5:
                raise ValueError("Less than half of the inference view is valid; check calibration, orientation and pitch")
        result = self.accelerator.apply(calibrated, maps=self.maps.xy[0])
        if result is not None:
            return result
        image = cv2.remap(calibrated, *self.maps.xy[0], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def close(self):
        self.accelerator.close()

    def backproject(self, landmarks):
        """Sample the same inverse map used for inference, including crop/resize.

        Returns raw pixel coordinates and a visibility mask. Invalid landmarks
        are never clamped onto the border and must not be drawn or used for pinch.
        """
        if self.maps is None:
            raise RuntimeError("prepare must run before backproject")
        points = np.asarray(landmarks, dtype=np.float32)
        coords = points[:, :2] * [self.view.width, self.view.height]
        finite = np.isfinite(coords).all(axis=1)
        coords = np.where(finite[:, None], coords, -1).astype(np.float32)
        x, y = coords[:, 0:1], coords[:, 1:2]
        valid = cv2.remap(self.maps.valid[0].astype(np.float32), x, y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).ravel() >= 0.999
        mapped = np.column_stack([
            cv2.remap(m, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=-1).ravel() for m in self.maps.xy[0]
        ])
        raw = undo_orientation(mapped, self.raw_size, self.channel)
        valid &= finite & (coords[:, 0] >= 0) & (coords[:, 0] < self.view.width - 1)
        valid &= (coords[:, 1] >= 0) & (coords[:, 1] < self.view.height - 1)
        raw[~valid] = np.nan
        return raw, valid
