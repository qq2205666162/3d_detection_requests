import os
import time
import numpy as np
import open3d as o3d

files = os.listdir("/home/yss/mark_plantform/python-scripts/3d_detection_requests/point_cloud/")

vis = o3d.visualization.Visualizer()
vis.create_window()
pointcloud = o3d.geometry.PointCloud()
to_reset = True
vis.add_geometry(pointcloud)
for f in files:
    print('f: ', f)
    pcd = o3d.io.read_point_cloud("/home/yss/mark_plantform/python-scripts/3d_detection_requests/point_cloud/" + f)
    pcd = np.asarray(pcd.points).reshape((-1, 3))
    pointcloud.points = o3d.utility.Vector3dVector(pcd)
    #vis.update_geometry()
    # 注意，如果使用的是open3d 0.8.0以后的版本，这句话应该改为下面格式
    vis.update_geometry(pointcloud)
    if to_reset:
        vis.reset_view_point(True)
        to_reset = False
    vis.poll_events()
    vis.update_renderer()
    time.sleep(1)