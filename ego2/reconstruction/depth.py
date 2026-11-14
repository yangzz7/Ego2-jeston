"""Rectified stereo matching with LR consistency and metric camera-frame points."""
import cv2
import numpy as np

from .calibration import calibration


def disparity_depth(left, right, valid_left, valid_right, focal, baseline, cfg):
    """right disparity uses the negative-disparity convention (x_right-x_left)."""
    h, w = left.shape
    y, x = np.indices((h, w))
    xr = np.rint(x - left).astype(np.int32)
    inside = (xr >= 0) & (xr < w)
    xr = np.clip(xr, 0, w - 1)
    valid = inside & valid_left & valid_right[y, xr] & (left > 0) & (right[y, xr] < 0)
    valid &= np.abs(left + right[y, xr]) <= cfg["lr_threshold"]
    z = focal * baseline / np.maximum(left, 1e-6)
    valid &= np.isfinite(z) & (z >= cfg["min_m"]) & (z <= cfg["max_m"])
    return np.where(valid, z, 0).astype(np.float32)


def local_points(depth, color, K, R_rect_camera, stride=3, voxel=0.015):
    y, x = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[y, x]
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]
    rect = np.column_stack(((x-K[0,2])*z/K[0,0], (y-K[1,2])*z/K[1,1], z))
    points = rect @ R_rect_camera  # row-vector equivalent of R.T @ p_rect
    colors = color[y, x, ::-1].copy()  # BGR -> RGB
    if len(points):
        _, idx = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
        points, colors = points[idx], colors[idx]
    return points.astype(np.float32), colors


class StereoDepth:
    def __init__(self, cfg):
        self.cfg = cfg["depth"]
        l, r, T = calibration(cfg)
        self.source_size = l.size
        size = (int(self.cfg["width"]), int(self.cfg["height"]))
        R1, R2, _, _, _ = cv2.fisheye.stereoRectify(
            l.K, l.D, r.K, r.D, l.size, T[:3,:3], T[:3,3],
            flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=size)
        # Keep OpenCV's stereo rotations, but choose a central 100-degree
        # pinhole crop. Its all-fisheye default wastes disparity on the periphery.
        fov = float(self.cfg.get("hfov_deg", 100))
        if not 40 <= fov <= 140:
            raise ValueError("Depth hfov_deg must be between 40 and 140 degrees")
        f = size[0] / (2*np.tan(np.deg2rad(fov)/2))
        self.K = np.array([[f,0,(size[0]-1)/2], [0,f,(size[1]-1)/2], [0,0,1]], np.float64)
        self.R1 = R1
        rect_t = R2 @ T[:3,3]
        self.baseline = float(-rect_t[0])
        if self.baseline <= 0 or np.linalg.norm(rect_t[1:]) > 1e-5:
            raise ValueError("Stereo order must produce a horizontal positive-disparity baseline")
        self.maps, self.valid = [], []
        for cam, R in ((l,R1), (r,R2)):
            mx, my = cv2.fisheye.initUndistortRectifyMap(cam.K, cam.D, R, self.K, size, cv2.CV_32FC1)
            self.maps.append((mx,my))
            self.valid.append((mx>=1)&(mx<cam.size[0]-2)&(my>=1)&(my<cam.size[1]-4))
        n, block = self.cfg["num_disparities"], self.cfg["block_size"]
        common = dict(numDisparities=n, blockSize=block, P1=8*block**2, P2=32*block**2,
                      disp12MaxDiff=1, preFilterCap=31, uniquenessRatio=12,
                      speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left_matcher = cv2.StereoSGBM_create(minDisparity=0, **common)
        self.right_matcher = cv2.StereoSGBM_create(minDisparity=-n, **common)

    def compute(self, left, right):
        if any(im is None or im.shape[:2][::-1] != self.source_size for im in (left,right)):
            raise ValueError(f"Expected two original RGB images of size {self.source_size}")
        a,b = [cv2.remap(im,*maps,cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
               for im,maps in zip((left,right),self.maps)]
        ga,gb = [cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in (a,b)]
        dl = self.left_matcher.compute(ga,gb).astype(np.float32)/16
        dr = self.right_matcher.compute(gb,ga).astype(np.float32)/16
        depth = disparity_depth(dl,dr,*self.valid,self.K[0,0],self.baseline,self.cfg)
        return depth,a

    def points(self, depth, color):
        return local_points(depth,color,self.K,self.R1,self.cfg["sample_stride"],self.cfg["voxel_m"])

    def metadata(self):
        return {"K": self.K.tolist(), "R_rect_camera": self.R1.tolist(), "baseline_m": self.baseline,
                "width": self.cfg["width"], "height": self.cfg["height"],
                "depth_unit": "metre", "camera": "cam1"}
