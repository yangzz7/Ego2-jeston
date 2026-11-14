"""Deterministic simulated textured plane for pipeline verification, not VIO."""
import cv2
import numpy as np

from .calibration import calibration
from .session import write_json


class PlaneScene:
    def __init__(self,cfg):
        l,r,self.T_right_left = calibration(cfg)
        self.cameras=(l,r)
        self.rays=[]
        rng=np.random.default_rng(42)
        self.texture=rng.integers(40,225,(1000,1400,3),dtype=np.uint8)
        self.texture=cv2.GaussianBlur(self.texture,(3,3),0.7)
        for i in range(35):
            cv2.circle(self.texture,tuple(rng.integers([0,0],[1400,1000])),int(rng.integers(6,45)),
                       tuple(int(v) for v in rng.integers(10,245,3)),2)
        for cam in self.cameras:
            y,x=np.indices((cam.size[1],cam.size[0]),dtype=np.float32)
            pixels=np.stack((x,y),-1).reshape(-1,1,2)
            undistorted=cv2.fisheye.undistortPoints(pixels,cam.K,cam.D).reshape(-1,2)
            self.rays.append(np.column_stack((undistorted,np.ones(len(undistorted)))))

    def render(self,pose):
        images=[]
        for cam,rays,T in zip(self.cameras,self.rays,(pose,pose@np.linalg.inv(self.T_right_left))):
            directions=rays@T[:3,:3].T
            good=np.isfinite(directions).all(axis=1)&(directions[:,2]>0)
            scale=(1.3-T[2,3])/np.maximum(directions[:,2],1e-8)
            p=T[:3,3]+directions*scale[:,None]
            uv=(p[:,:2]+[1.75,1.25])*400
            uv[~good]=-1
            uv=np.clip(uv,-10000,10000).astype(np.float32).reshape(cam.size[1],cam.size[0],2)
            images.append(cv2.remap(self.texture,uv,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT))
        return images


def run_demo(cfg,args,session,dense,viewer):
    print("[demo] 合成的 1.3 m 平面与已知位姿，仅验证深度/显示/导出，不代表真实 VIO 效果。",flush=True)
    scene=PlaneScene(cfg)
    snapshot={"version":1,"mode":"demo","state":2,"active_map":"0_0","final":False,
              "maps":[{"id":"0_0","imu_initialized":True,"revision":0,"keyframes":[]}]}
    for i in range(args.frames):
        runtime=getattr(args,'recording_runtime',None)
        if runtime:runtime.poll()
        pose=np.eye(4)
        pose[0,3]=i*0.015
        images=scene.render(pose)
        stamp=1+i/10
        key=str(round(stamp*1e6))
        for name,im in zip(("cam1","cam0"),images):
            cv2.imwrite(str(session/"keyframes"/(key+f"_{name}.jpg")),im,[cv2.IMWRITE_JPEG_QUALITY,95])
        snapshot["maps"][0]["keyframes"].append(dict(id=i,timestamp=stamp,image_key=key,
            image_available=True,T_world_camera=pose.tolist()))
        if viewer:
            viewer.rgb(images,stamp-1)
        dense.update(snapshot,stamp-1,budget=1)
    snapshot["final"]=True
    write_json(session/"maps.json",snapshot)
    dense.update(snapshot,(args.frames-1)/10,budget=None)
    print(f'[{"cache" if getattr(args,"recording_runtime",None) else "saved"}] {session}',flush=True)
    return session
