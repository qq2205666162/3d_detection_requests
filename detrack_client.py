#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import requests
import os
import threading
from flask import Flask,request
from InterfaceAPI import *
import open3d as o3d
import load_pcd
import io

accept_data = None
app = Flask(__name__)

# 监听,用于接收返回值
@app.route("/ourself/", methods=["GET", "POST"])
def data_accept():
    global accept_data
    comment = json.loads(request.data)
    accept_data = comment
    # print('posResult: ', accept_data)
    state = 200 if comment else 400
    return json.dumps({"err_code": state})

def worker():
    app.run(host='10.10.31.70',port=26001)      # host表示本机地址，port表示返回参数的监听端口


def get_json_GT_result(path):
    labels = []
    categray = []
    with open(path, 'r') as f:
        data = json.load(f)
        
        for boxes in data['objects']:
            label = []
            value = boxes['contour']['center3D']['x']
            if type(value) == float or type(value) == int:
                label.append(value)
            else:
                label.append(value[0])
            
            value = boxes['contour']['center3D']['y']
            if type(value) == float or type(value) == int:
                label.append(value)
            else:
                label.append(value[0])

            value = boxes['contour']['center3D']['z']
            if type(value) == float or type(value) == int:
                label.append(value)
            else:
                label.append(value[0])

            value = boxes['contour']['size3D']['x']
            if type(value) == float or type(value) == int:
                label.append(value)
            else:
                label.append(value[0])

            value = boxes['contour']['size3D']['y']
            if type(value) == float or type(value) == int:
                label.append(value)
            else:
                label.append(value[0])

            value = boxes['contour']['size3D']['z']
            if type(value) == float or type(value) == int:
                label.append(value)
            else:
                label.append(value[0])

            value = boxes['contour']['rotation3D']['z']
            if type(value) == float or type(value) == int:
                label.append(value)
            else:
                label.append(value[0])  
            
            labels.append(label)
            categray.append(boxes['className'])
    return labels, categray


def get_all_data(path_labels):
    label_all = []
    for i in range(10):
        name_lab = str(i + 60) + '.json'
        lab_pa = os.path.join(path_labels, name_lab)
        labels, categray = get_json_GT_result(lab_pa)
        labels_car = []
        i = 0
        for label in labels:
            cat = categray[i]
            if cat == '小车' or cat == '大车' or cat == '超大车':
               labels_car.append(label)
            i = i + 1
        label_all.append(labels_car)
    return label_all


def get_all_urls(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def save_result_and_pointcloud(frame_name, detection, pointCloudFile):

    r = requests.get(pointCloudFile, allow_redirects=True)
    pc = load_pcd.PointCloud(io.BytesIO(r.content)).numpy(fields=['x', 'y', 'z', 'intensity', 'i'])
    point_cloud = o3d.geometry.PointCloud()
    for i in range(len(pc)):
        point_cloud.points.append([pc[i][0], pc[i][1], pc[i][2]])
    o3d.io.write_point_cloud("point_cloud/" + frame_name + ".pcd", point_cloud, write_ascii=True)

    with open("detection/" + frame_name + '.json', 'w') as f:
        json.dump(json.loads(detection), f, indent = 4)


def predict_json_file(json_file = '/home/yss/下载/31282_1845800063629897728.json'):
    new = threading.Thread(target=worker, name='new')
    new.start()
    time.sleep(3)
    url = f"http://10.3.0.171:333/AILabel/" 
    urls= get_all_urls(json_file)

    print('size of urls:', len(urls['datas']))
    print('urls: ', urls)
    #fordata = {'datas': [{'id': 0, 'pointCloudFile': urls}], 'params': {}}
    res = requests.post(url, data=json.dumps(urls, indent = 4), timeout=300)
    print('res-----------------: ', res)
    print('res.text-----------------: ', res.text)
    with open('detrack_result.json', 'w') as f:
        json.dump(json.loads(res.text), f , indent = 4)

if __name__ == '__main__':
    predict_json_file()
    exit(0)