import json
import math
import open3d as o3d
import numpy as np
import os
import cv2
import time

def listPathAllfiles(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result

def get_labels(path = '/home/yss/mark_plantform/python-scripts/3d_detection_requests/tst.json'):
    labels = []
    with open(path, 'r') as f:
        data = json.load(f)
        for box in data['data'][0]['classes']:
            label = []
            label.append(float(box['x']))
            label.append(float(box['y']))
            label.append(float(box['z']))
            label.append(float(box['dx']))
            label.append(float(box['dy']))
            label.append(float(box['dz']))
            label.append(float(box['rotZ']))

            confidence = float(box['confidence'])
            if confidence > 0.6:
                labels.append(label)
            #print('labels: ', label)
    #print('labels: ', labels)
    return labels

def draw_arrow(conners, r, g, b):
    x_center = (conners[0][0] + conners[1][0] + conners[2][0] + conners[3][0]) / 4.
    y_center = (conners[0][1] + conners[1][1] + conners[2][1] + conners[3][1]) / 4.
    z_center = (conners[0][2] + conners[1][2] + conners[2][2] + conners[3][2]) / 4.
    x_center2 = (conners[2][0] + conners[3][0]) / 2. + .5*((conners[2][0] + conners[3][0]) / 2. - x_center)
    y_center2 = (conners[2][1] + conners[3][1]) / 2. + .5*((conners[2][1] + conners[3][1]) / 2. - y_center)
    z_center2 = (conners[2][2] + conners[3][2]) / 2.

    polygon_points = np.array([[x_center, y_center, z_center], [x_center2, y_center2, z_center2]])
    lines = [[0, 1], [1, 0]]
    color = [[r, g, b] for i in range(len(lines))]
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color) 
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    return lines_pcd

def draw_cube(conners, r, g, b):
    polygon_points = conners
    lines = [[0, 1], [1, 2], [2, 3],[3, 0],[0, 4], [1, 5],[2, 6],[3, 7], [4, 5],[5, 6],[6, 7],[7, 4]]
    color = [[r, g, b] for i in range(len(lines))]
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color) 
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    return lines_pcd

def get_conners(label):
    conners = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],[0., 0. ,0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    conners[0][0] = label[0] - label[3]/2
    conners[0][1] = label[1] + label[4]/2
    conners[0][2] = label[2] + label[5]/2

    conners[1][0] = label[0] - label[3]/2
    conners[1][1] = label[1] - label[4]/2
    conners[1][2] = label[2] + label[5]/2

    conners[2][0] = label[0] + label[3]/2
    conners[2][1] = label[1] - label[4]/2
    conners[2][2] = label[2] + label[5]/2

    conners[3][0] = label[0] + label[3]/2
    conners[3][1] = label[1] + label[4]/2
    conners[3][2] = label[2] + label[5]/2

    conners[4][0] = label[0] - label[3]/2
    conners[4][1] = label[1] + label[4]/2
    conners[4][2] = label[2] - label[5]/2

    conners[5][0] = label[0] - label[3]/2
    conners[5][1] = label[1] - label[4]/2
    conners[5][2] = label[2] - label[5]/2

    conners[6][0] = label[0] + label[3]/2
    conners[6][1] = label[1] - label[4]/2
    conners[6][2] = label[2] - label[5]/2

    conners[7][0] = label[0] + label[3]/2
    conners[7][1] = label[1] + label[4]/2
    conners[7][2] = label[2] - label[5]/2

    i = 0
    for conner in conners:
        exchange_x = (conner[0] - label[0]) * math.cos(label[6]) - (conner[1] - label[1]) * math.sin(label[6]) + label[0]
        exchange_y = (conner[0] - label[0]) * math.sin(label[6]) + (conner[1] - label[1]) * math.cos(label[6]) + label[1]
        conners[i][0] = exchange_x
        conners[i][1] = exchange_y
        i += 1
    return conners

def visual(path_pcd, labels):
    cloud = o3d.io.read_point_cloud(path_pcd)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='fffff')
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0]) 
    opt.point_size = 1 
    vis.add_geometry(cloud)
    for label in labels:
        print('label: ', label)
        conners = get_conners(label)
        lines = draw_cube(conners, 1., 0., 0.)
        arrow = draw_arrow(conners, 0, 1., 1.)
        vis.add_geometry(lines)
        vis.add_geometry(arrow)
    vis.run()

def make_vadio(path_pcd, path_detection):
    pcd_files_ = listPathAllfiles(path_pcd)
    detection_files_ = listPathAllfiles(path_detection)

    pcd_files = pcd_files_
    pcd_files_use = []
    for i in range(len(pcd_files)):
        for j in range(len(pcd_files_)):
            index2 = pcd_files_[j].find('_640_')
            index = pcd_files_[j].find('.')
            
            if int(pcd_files_[j][index2+5 : index]) == i:
                print('i: ', i)
                print('pcd_files_[j][index2+5 : index]: ', pcd_files_[j][index2+5 : index])
                pcd_files_use.append(pcd_files_[j])
                pcd_files[i] = pcd_files_[j]

    detection_files = detection_files_
    detection_files_use = []
    for i in range(len(detection_files)):
        for j in range(len(detection_files_)):
            index2 = detection_files_[j].find('_640_')
            index = detection_files_[j].find('.')
            if int(detection_files_[j][index2+5 : index]) == i:
                detection_files[i] = detection_files_[j]
                detection_files_use.append(detection_files_[j])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='fffff')
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0]) 
    opt.point_size = 1
    pointcloud = o3d.geometry.PointCloud()
    vis.add_geometry(pointcloud)
    
    for i in range(len(pcd_files_use)):
        print('pcd_files_use[i]: ', pcd_files_use[i])
        pcd = o3d.io.read_point_cloud(pcd_files_use[i])
        pcd = np.asarray(pcd.points).reshape((-1, 3))
        labels = get_labels(detection_files_use[i])

        pointcloud.points = o3d.utility.Vector3dVector(pcd)
        for label in labels:
            #print('label: ', label)
            conners = get_conners(label)
            lines = draw_cube(conners, 1., 0., 0.)
            print(lines.points)
            arrow = draw_arrow(conners, 0, 1., 1.)
            vis.add_geometry(lines)
            vis.add_geometry(arrow)

        vis.update_geometry(pointcloud)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1)

if __name__ == '__main__':

    make_vadio('/home/yss/mark_plantform/python-scripts/3d_detection_requests/point_cloud', '/home/yss/mark_plantform/python-scripts/3d_detection_requests/detection')
    exit(0)

    labels = get_labels('/home/yss/mark_plantform/python-scripts/3d_detection_requests/detection/1715761798100000000_640_0.json')
    visual('/home/yss/mark_plantform/python-scripts/3d_detection_requests/point_cloud/1715761798100000000_640_0.pcd', labels)