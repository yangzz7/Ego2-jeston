"""Offline per-map cloud/TSDF export. Can run with a separate Open3D Python."""
from pathlib import Path
import argparse
import json
import numpy as np


def export(session, kind="both", voxel=0.02):
    import open3d as o3d
    session=Path(session).resolve()
    snapshot=json.loads((session/"maps.json").read_text())
    rect=json.loads((session/"rectification.json").read_text())
    info=json.loads((session/"session.json").read_text())
    if not snapshot.get("final"):
        raise ValueError("Session has no final map snapshot. Stop acquisition cleanly before export.")
    K=np.array(rect["K"])
    R=np.array(rect["R_rect_camera"])
    T_camera_rect=np.eye(4); T_camera_rect[:3,:3]=R.T
    intrinsic=o3d.camera.PinholeCameraIntrinsic(rect["width"],rect["height"],K[0,0],K[1,1],K[0,2],K[1,2])
    out=session/"exports"; out.mkdir(exist_ok=True)
    results=[]
    for map_ in snapshot["maps"]:
        if not map_["imu_initialized"] and snapshot.get("mode") != "stereo":
            continue
        if snapshot.get("version",1) >= 2 and snapshot.get("mode") not in ("stereo","demo"):
            if not map_.get("imu_initialization_complete",False):
                continue
        volume=o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel,sdf_trunc=voxel*5,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        cloud=o3d.geometry.PointCloud()
        count=0
        for kf in sorted(map_["keyframes"],key=lambda k:k["timestamp"]):
            if not kf.get("image_available"):
                continue
            cache=session/"depth"/(kf["image_key"]+".npz")
            if not cache.exists():
                raise ValueError(f"Missing depth cache: {cache}; run the export command through ego2.reconstruction")
            with np.load(cache,allow_pickle=False) as data:
                depth=data["depth"].astype(np.float32)
                rgb=np.ascontiguousarray(data["color"][:,:,::-1])
            if not np.any(depth>0):
                continue
            pose=np.asarray(kf["T_world_camera"])@T_camera_rect
            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb),o3d.geometry.Image(depth),depth_scale=1.0,
                depth_trunc=float(info["config"]["depth"]["max_m"])+0.01,convert_rgb_to_intensity=False)
            if kind in ("mesh","both"):
                volume.integrate(rgbd,intrinsic,np.linalg.inv(pose))
            if kind in ("points","both"):
                part=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,intrinsic)
                part.transform(pose)
                cloud+=part.voxel_down_sample(voxel)
                # Bound intermediate cloud growth even for long sessions.
                if count%20 == 0:
                    cloud=cloud.voxel_down_sample(voxel)
            count+=1
        if count==0:
            continue
        record={"map":map_["id"],"keyframes":count}
        if kind in ("points","both"):
            cloud=cloud.voxel_down_sample(voxel)
            path=out/("map_"+map_["id"]+"_points.ply")
            if not o3d.io.write_point_cloud(str(path),cloud):
                raise RuntimeError(f"Failed to write {path}")
            record.update(points=len(cloud.points),points_file=str(path))
        if kind in ("mesh","both"):
            mesh=volume.extract_triangle_mesh(); mesh.compute_vertex_normals()
            path=out/("map_"+map_["id"]+"_mesh.ply")
            if len(mesh.triangles)==0:
                raise ValueError("No surface triangles; collect more overlapping depth observations or reduce voxel size")
            if not o3d.io.write_triangle_mesh(str(path),mesh):
                raise RuntimeError(f"Failed to write {path}")
            record.update(triangles=len(mesh.triangles),mesh_file=str(path))
        results.append(record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    if not results:
        raise ValueError("No initialized map with valid depth. Move the camera to initialize VIO and scan textured surfaces.")
    (out/"export.json").write_text(json.dumps({"voxel_m":voxel,"maps":results},indent=2))
    return results


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--session",required=True)
    parser.add_argument("--format",choices=("points","mesh","both"),default="both")
    parser.add_argument("--voxel",type=float,default=0.02)
    args=parser.parse_args()
    if not np.isfinite(args.voxel) or args.voxel<=0:
        parser.error("--voxel must be positive")
    export(args.session,args.format,args.voxel)
