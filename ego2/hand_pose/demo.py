"""Synthetic fisheye images of moving printed tags; run the REAL detector/solver."""
import time
import cv2
import numpy as np
from ..common.sources import Pair
from ..common.geometry import project
from .config import dictionary
from .pose import corners
from .bundle import marker_ids, synthetic_bundle


class DemoSource:
    kind = "demo"

    def __init__(self, cfg, cameras):
        self.cfg, self.cameras = cfg, cameras
        self.count = 0
        self.fps = float(cfg["input"]["fps"])
        self.truth = {}
        self.rays = {}
        self.textures = {}
        for target in cfg["targets"]:
            if "ids" in target and "T_wrist_marker" not in target:
                target["T_wrist_marker"] = synthetic_bundle(target)
            for marker_id in marker_ids(target):
                self.make_texture(marker_id)

    def make_texture(self, marker_id):
        cfg = self.cfg
        cells = dictionary(cfg["dictionary"]).markerSize + 2*cfg["border_bits"]
        tag = cv2.aruco.generateImageMarker(dictionary(cfg["dictionary"]), marker_id, cells*32,
                                           borderBits=cfg["border_bits"])
        self.textures[marker_id] = cv2.copyMakeBorder(tag, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)

    def open(self):
        self.next_time = time.monotonic()
        for channel in self.cfg["channels"]:
            cam = self.cameras[channel["camera"]]
            width, height = cam.size
            yy, xx = np.mgrid[:height, :width]
            pixels = np.stack([xx, yy], axis=-1).astype(np.float64)
            normalized = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), cam.K, cam.D).reshape(height, width, 2)
            rays = np.dstack([normalized, np.ones((height, width))])
            u, v, valid = project(cam, rays)
            # OpenCV clips extreme fisheye angles; never render aliased copies there.
            valid &= (np.abs(u-xx) < 0.1) & (np.abs(v-yy) < 0.1)
            self.rays[cam.name] = (rays, valid)

    def poses(self, stamp):
        result = {}
        for i, target in enumerate(self.cfg["targets"]):
            phase = stamp*1.6 + i*0.7
            size_factor = target["size_m"] / 0.0225
            pose = np.eye(4)
            pose[:3, 3] = np.array([(-0.04 if i == 0 else 0.055) + 0.014*np.cos(phase),
                                   0.045 + 0.02*np.sin(phase), 0.22 + i*0.025 + 0.015*np.sin(phase*0.7)]) * size_factor
            pose[:3, :3] = cv2.Rodrigues(np.array([0.22*np.sin(phase), 0.22*np.cos(phase), 0.15*np.sin(phase)]))[0] @ np.diag([1., -1., -1.])
            if "ids" in target:
                pose[:3, 3][0] += -.018 if i == 0 else .018
                pose[:3, 3][2] += .015
                pose[:3, :3] = pose[:3, :3] @ cv2.Rodrigues(np.array([0., stamp*2.2, 0.]))[0]
            result[target["name"]] = pose
        return result

    def render(self, camera, poses):
        rays, valid = self.rays[camera.name]
        width, height = camera.size
        image = np.full((height, width), 160, np.uint8)
        depth = np.full((height, width), np.inf)
        root_from_head = np.linalg.inv(self.cameras[self.cfg["reference"]].T_from_root)
        items = []
        for target in self.cfg["targets"]:
            for marker_id in marker_ids(target):
                offset = np.asarray(target["T_wrist_marker"][str(marker_id)]) if "ids" in target else np.eye(4)
                items.append((target, marker_id, poses[target["name"]] @ offset))
        for target, marker_id, pose in items:
            t = camera.T_from_root @ root_from_head @ pose
            if t[:3, 2] @ t[:3, 3] >= 0:
                continue
            texture = self.textures[marker_id]
            cells = dictionary(self.cfg["dictionary"]).markerSize + 2*self.cfg["border_bits"]
            extent = target["size_m"] * (cells+2)/cells
            xyz = corners(extent) @ t[:3, :3].T + t[:3, 3]
            u, v, ok = project(camera, xyz)
            if not ok.all():
                continue
            x0, x1 = max(0, int(np.floor(u.min()))-3), min(width, int(np.ceil(u.max()))+4)
            y0, y1 = max(0, int(np.floor(v.min()))-3), min(height, int(np.ceil(v.max()))+4)
            if x1 <= x0 or y1 <= y0:
                continue
            local_rays = rays[y0:y1, x0:x1] @ t[:3, :3]
            origin = -t[:3, :3].T @ t[:3, 3]
            distances = -origin[2] / np.where(np.abs(local_rays[..., 2]) > 1e-8, local_rays[..., 2], 1e-8)
            local = origin + distances[..., None]*local_rays
            mask = ((np.abs(local[..., 0]) <= extent/2) & (np.abs(local[..., 1]) <= extent/2) &
                    (distances > 0) & (distances < depth[y0:y1, x0:x1]) & valid[y0:y1, x0:x1])
            mx = ((local[..., 0]/extent + 0.5)*texture.shape[1] - 0.5).astype(np.float32)
            my = ((0.5 - local[..., 1]/extent)*texture.shape[0] - 0.5).astype(np.float32)
            patch = cv2.remap(texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            image[y0:y1, x0:x1][mask] = patch[mask]
            depth[y0:y1, x0:x1][mask] = distances[mask]
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def read(self, stop):
        if stop.wait(max(0, self.next_time - time.monotonic())):
            return None
        self.next_time = max(time.monotonic(), self.next_time + 1/self.fps)
        stamp = self.count / self.fps
        self.truth = self.poses(stamp)
        images = tuple(self.render(self.cameras[ch["camera"]], self.truth) for ch in self.cfg["channels"])
        self.count += 1
        return Pair(images, stamp, time.monotonic(), self.count)

    def close(self):
        pass
