"""Virtual perspective view -> calibrated equidistant ray mapping."""
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class View:
    width: int = 1280
    height: int = 720
    hfov: float = 110.0
    yaw: float = 0.0
    pitch: float = 0.0
    distance: float = 2.0  # 0 = infinity
    blend_width: int = 16
    coverage_mode: str = "free"
    min_coverage: float = 0.999

    def __post_init__(self):
        if any(not isinstance(x,int) for x in (self.width,self.height,self.blend_width)):
            raise ValueError("Output width, height and blend_width must be integers")
        vals = [self.hfov, self.yaw, self.pitch, self.distance, self.min_coverage]
        if not all(math.isfinite(x) for x in vals):
            raise ValueError("View parameters must be finite")
        if not 32 <= self.width <= 4096 or not 32 <= self.height <= 2160:
            raise ValueError("Output size must be within 32..4096 x 32..2160")
        if not 30 <= self.hfov <= 160 or not -180 <= self.yaw <= 180 or not -89 <= self.pitch <= 89:
            raise ValueError("HFOV 30..160, yaw -180..180, pitch -89..89 required")
        if self.distance < 0 or not 0 <= self.blend_width <= 128:
            raise ValueError("distance must be >=0; blend_width 0..128")
        if self.coverage_mode not in ("free", "full") or not 0 < self.min_coverage <= 1:
            raise ValueError("Invalid coverage policy")

    @property
    def vfov(self):
        return math.degrees(2*math.atan(self.height/self.width * math.tan(math.radians(self.hfov/2))))

    def geometry_key(self):
        return self.width, self.height, self.hfov, self.yaw, self.pitch, self.distance


def rotation(yaw, pitch):
    """Optical coordinates: x right, y down, z forward. Positive pitch looks up."""
    y, p = np.deg2rad([yaw, pitch])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return ry @ rx


def project(camera, xyz):
    """Extended equidistant projection, with a monotonic angular-domain check.

    Unlike a pinhole x/z implementation, atan2 preserves the sign/domain of
    rays beyond 90 degrees. This is needed when turning a fisheye viewport.
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    t2 = theta*theta
    k1, k2, k3, k4 = camera.D
    rho = theta*(1+t2*(k1+t2*(k2+t2*(k3+t2*k4))))
    scale = np.divide(rho, r, out=np.zeros_like(r), where=r > 1e-12)
    xn, yn = x*scale, y*scale
    u = camera.K[0, 0]*xn + camera.K[0, 1]*yn + camera.K[0, 2]
    v = camera.K[1, 1]*yn + camera.K[1, 2]
    valid = (theta < camera.theta_limit) & (rho >= 0) & (np.linalg.norm(xyz, axis=-1) > 1e-9)
    return u, v, valid & np.isfinite(u) & np.isfinite(v)


@dataclass
class Maps:
    view: View
    xy: list
    valid: list
    sizes: tuple

    @property
    def coverage(self):
        return float(np.mean(self.valid[0] | self.valid[1]))

    @property
    def overlap(self):
        return float(np.mean(self.valid[0] & self.valid[1]))


class Mapper:
    def __init__(self, cameras, channels, reference):
        # Algorithm uses reference first, independent of input lane order.
        self.order = sorted(range(2), key=lambda i: channels[i]["camera"] != reference)
        self.channels = [channels[i] for i in self.order]
        self.cameras = [cameras[c["camera"]] for c in self.channels]
        base_inverse = np.linalg.inv(cameras[reference].T_from_root)
        self.transforms = [c.T_from_root @ base_inverse for c in self.cameras]
        self.masks = []
        for c, ch in zip(self.cameras, self.channels):
            mask = np.full((c.size[1], c.size[0]), 255, np.uint8)
            if ch.get("valid_mask"):
                mask = cv2.imread(ch["valid_mask"], cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (c.size[1], c.size[0]):
                    raise ValueError(f"valid_mask must have calibration resolution: {c.name}")
                mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            rows = int(ch.get("invalid_bottom_rows", 0))
            if not 0 <= rows < c.size[1]:
                raise ValueError("invalid_bottom_rows outside image")
            if rows:
                mask[-rows:] = 0
            self.masks.append(mask)

    def build(self, view, sizes):
        h, w = view.height, view.width
        f = w / (2*np.tan(np.deg2rad(view.hfov/2)))
        yy, xx = np.mgrid[:h, :w].astype(np.float64)
        rays = np.stack([(xx-(w-1)/2)/f, (yy-(h-1)/2)/f, np.ones_like(xx)], axis=-1)
        rays = rays @ rotation(view.yaw, view.pitch).T
        if view.distance:
            rays *= view.distance
        coords, valid = [], []
        for cam, ch, transform, mask, size in zip(self.cameras, self.channels, self.transforms, self.masks, sizes):
            xyz = rays @ transform[:3, :3].T
            if view.distance:
                xyz += transform[:3, 3]
            u, v, ok = project(cam, xyz)
            native_ok = cv2.remap(mask, u.astype(np.float32), v.astype(np.float32),
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) == 255
            crop = ch.get("calibration_crop", [0, 0, *cam.size])
            x, y, cw, chh = map(float, crop)
            if cw <= 0 or chh <= 0 or x < 0 or y < 0 or x+cw > cam.size[0] or y+chh > cam.size[1]:
                raise ValueError(f"Invalid calibration_crop: {cam.name}")
            iw, ih = size
            if not ch.get("allow_anisotropic_resize", False) and abs((iw/ih)/(cw/chh)-1) > 0.002:
                raise ValueError(f"{cam.name}: input {iw}x{ih} does not match calibration aspect ratio; declare calibration_crop or allow_anisotropic_resize")
            # OpenCV resize pixel-centre convention, also accounting for crop.
            mx = ((u-x+0.5)*(iw/cw)-0.5).astype(np.float32)
            my = ((v-y+0.5)*(ih/chh)-0.5).astype(np.float32)
            ok &= native_ok & (mx >= 0) & (my >= 0) & (mx < iw-1) & (my < ih-1)
            mx[~ok], my[~ok] = -1, -1
            coords.append((mx, my))
            valid.append(ok)
        return Maps(view, coords, valid, tuple(sizes))

    def fit(self, view, sizes):
        """Keep the requested direction, shrink FOV; return None if impossible.

        Coarse candidates only propose a size. Full resolution verifies every
        accepted result (narrow invalid masks must not disappear at low res).
        """
        candidates = list(np.arange(view.hfov, 29.99, -2))
        if not candidates or candidates[-1] > 30:
            candidates.append(30.0)
        for angle in candidates:
            candidate = replace(view, hfov=float(angle))
            small = replace(candidate, width=320, height=max(32, round(320*view.height/view.width)))
            if self.build(small, sizes).coverage >= view.min_coverage:
                maps = self.build(candidate, sizes)
                if maps.coverage >= view.min_coverage:
                    return maps
        return None


def checkerboard(size):
    w, h = size
    y, x = np.mgrid[:h, :w]
    gray = np.where((x//16+y//16) % 2, 55, 85).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def validity_preview(maps):
    out = checkerboard((maps.view.width, maps.view.height))
    a, b = maps.valid
    out[a & ~b] = (235, 130, 40)  # BGR blue
    out[b & ~a] = (40, 155, 245)
    out[a & b] = (65, 175, 65)
    return out


def source_overlay(image, maps, index, weight):
    """Splat the full valid viewport, not just its four corners, onto input."""
    ih, iw = image.shape[:2]
    thumb_w = 480
    thumb_h = max(1, round(ih*thumb_w/iw))
    out = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    # The overlay is a 480-pixel thumbnail; dense full-output splats needlessly
    # dominate display time at 720p/1080p. Still sample the complete viewport.
    stride = max(1, maps.view.width//480)
    mx, my = (m[::stride, ::stride] for m in maps.xy[index])
    valid = maps.valid[index][::stride, ::stride]
    weight = weight[::stride, ::stride]
    selected = (weight > 0.5) if index == 0 else (weight < 0.5)
    x = np.clip(np.rint((mx[valid]+0.5)*thumb_w/iw-0.5).astype(int), 0, thumb_w-1)
    y = np.clip(np.rint((my[valid]+0.5)*thumb_h/ih-0.5).astype(int), 0, thumb_h-1)
    footprint = np.zeros((thumb_h, thumb_w), np.uint8)
    used = footprint.copy()
    footprint[y, x] = 255
    used[y[selected[valid]], x[selected[valid]]] = 255
    kernel = np.ones((3, 3), np.uint8)
    footprint = cv2.morphologyEx(footprint, cv2.MORPH_CLOSE, kernel)
    used = cv2.morphologyEx(used, cv2.MORPH_CLOSE, kernel)
    color = np.array([255, 140, 40] if index == 0 else [40, 170, 255])
    flag = used > 0
    out[flag] = (0.65*out[flag]+0.35*color).astype(np.uint8)
    contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 1)
    return out
