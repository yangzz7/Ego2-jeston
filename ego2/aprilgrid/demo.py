"""Render a moving physical AprilGrid into both calibrated fisheye cameras."""
import time
import cv2
import numpy as np
from ..common.geometry import project
from ..common.sources import Pair


class DemoSource:
    kind = "demo"

    def __init__(self, cfg, cameras, board):
        self.cfg, self.cameras, self.board = cfg, cameras, board
        self.count = 0
        self.fps = float(cfg["input"]["fps"])
        self.truth = None
        self.rays = {}
        # Use a physically scaled page, including white gutters and margins.
        scale = 4000
        size = np.ceil(board.page_size*scale).astype(int)
        self.texture = np.full((size[1], size[0]), 255, np.uint8)
        for marker_id, centre in board.centres.items():
            tag_size = round(board.size*scale)
            tag = cv2.aruco.generateImageMarker(board.dictionary, marker_id, tag_size, borderBits=board.border)
            tag = np.rot90(tag, -(board.rotation//90))
            x = round((centre[0] + board.page_size[0]/2-board.size/2)*scale)
            y = round((board.page_size[1]/2-centre[1]-board.size/2)*scale)
            self.texture[y:y+tag_size, x:x+tag_size] = tag
        if board.data.get("corner_squares", False):
            gap = board.pitch-board.size
            for row in range(board.rows+1):
                for col in range(board.cols+1):
                    x = (col-board.cols/2)*board.pitch
                    y = (row-board.rows/2)*board.pitch
                    x0 = round((x+board.page_size[0]/2-gap/2)*scale)
                    y0 = round((y+board.page_size[1]/2-gap/2)*scale)
                    x1 = round((x+board.page_size[0]/2+gap/2)*scale)
                    y1 = round((y+board.page_size[1]/2+gap/2)*scale)
                    self.texture[max(0,y0):min(size[1],y1), max(0,x0):min(size[0],x1)] = 0
        ref_inverse = np.linalg.inv(cameras[cfg["reference"]].T_from_root)
        self.transforms = {name: cam.T_from_root @ ref_inverse for name, cam in cameras.items()}

    def open(self):
        self.next_time = time.monotonic()
        for ch in self.cfg["channels"]:
            cam = self.cameras[ch["camera"]]
            w, h = cam.size
            yy, xx = np.mgrid[:h, :w]
            # Invert the angular polynomial, preserving rays beyond 90 degrees.
            yn = (yy-cam.K[1, 2])/cam.K[1, 1]
            xn = (xx-cam.K[0, 2]-cam.K[0, 1]*yn)/cam.K[0, 0]
            rho = np.hypot(xn, yn)
            lo, hi = np.zeros_like(rho), np.full_like(rho, cam.theta_limit)
            a, b, c, d = cam.D
            def radial(theta):
                t2 = theta*theta
                return theta*(1+t2*(a+t2*(b+t2*(c+t2*d))))
            for _ in range(30):
                mid = (lo+hi)/2
                select = radial(mid) < rho
                lo, hi = np.where(select, mid, lo), np.where(select, hi, mid)
            theta = (lo+hi)/2
            scale = np.sin(theta)/np.maximum(rho, 1e-12)
            self.rays[cam.name] = (np.stack([xn*scale, yn*scale, np.cos(theta)], axis=-1),
                                    rho < radial(cam.theta_limit))

    def pose_at(self, stamp):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([.18, .18*np.cos(stamp), .06*np.sin(stamp)]))[0] @ np.diag([1., -1., -1.])
        pose[:3, 3] = [.035*np.sin(stamp), .02*np.cos(stamp), .48+.03*np.sin(stamp)]
        return pose

    def render(self, cam, pose):
        rays, valid = self.rays[cam.name]
        t = self.transforms[cam.name] @ pose
        local_rays = rays @ t[:3, :3]
        origin = -t[:3, :3].T @ t[:3, 3]
        denom = local_rays[..., 2]
        distance = -origin[2]/np.where(np.abs(denom) > 1e-9, denom, 1e-9)
        local = origin + distance[..., None]*local_rays
        width, height = self.board.page_size
        mask = valid & (distance > 0) & (np.abs(local[..., 0]) <= width/2) & (np.abs(local[..., 1]) <= height/2)
        mx = ((local[..., 0]/width+.5)*self.texture.shape[1]-.5).astype(np.float32)
        my = ((.5-local[..., 1]/height)*self.texture.shape[0]-.5).astype(np.float32)
        image = cv2.remap(self.texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=180)
        image[~mask] = 65
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def read(self, stop):
        if stop.wait(max(0, self.next_time-time.monotonic())):
            return None
        self.next_time = max(time.monotonic(), self.next_time+1/self.fps)
        stamp = self.count/self.fps
        self.truth = self.pose_at(stamp)
        images = tuple(self.render(self.cameras[ch["camera"]], self.truth) for ch in self.cfg["channels"])
        self.count += 1
        return Pair(images, stamp, time.monotonic(), self.count)

    def close(self):
        pass
