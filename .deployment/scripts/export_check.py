"""Test APIs used by legacy reconstruction export without a display/camera."""
import tempfile
from pathlib import Path
import numpy as np
import open3d as o3d

camera=o3d.camera.PinholeCameraIntrinsic(32,32,25,25,16,16)
rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(np.zeros((32,32,3),np.uint8)),
    o3d.geometry.Image(np.ones((32,32),np.float32)),
    depth_scale=1.0,depth_trunc=2.0,convert_rgb_to_intensity=False)
cloud=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,camera)
cloud.transform(np.eye(4));cloud=cloud.voxel_down_sample(.02)
volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.02,sdf_trunc=.1,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
volume.integrate(rgbd,camera,np.eye(4))
mesh=volume.extract_triangle_mesh();mesh.compute_vertex_normals()
with tempfile.TemporaryDirectory(prefix='ego2-open3d-check-') as temp:
    assert o3d.io.write_point_cloud(str(Path(temp)/'points.ply'),cloud)
print('Open3D',o3d.__version__,'legacy export API smoke test passed; real dataset export still pending.')
