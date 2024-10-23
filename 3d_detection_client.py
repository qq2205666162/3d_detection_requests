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
    urls = []
    frame_names = []
    with open(json_file, 'r') as f:
        data = json.load(f)
        for e in data['datas']:
            frame_names.append(e['frameName'])
            urls.append(e['pointCloudFile'])
    return urls, frame_names


def save_result_and_pointcloud(frame_name, detection, pointCloudFile):

    r = requests.get(pointCloudFile, allow_redirects=True)
    pc = load_pcd.PointCloud(io.BytesIO(r.content)).numpy(fields=['x', 'y', 'z', 'intensity', 'i'])
    point_cloud = o3d.geometry.PointCloud()
    for i in range(len(pc)):
        point_cloud.points.append([pc[i][0], pc[i][1], pc[i][2]])
    o3d.io.write_point_cloud("point_cloud/" + frame_name + ".pcd", point_cloud, write_ascii=True)

    with open("detection/" + frame_name + '.json', 'w') as f:
        json.dump(json.loads(detection), f, indent = 4)


def predict_json_file(json_file = '/home/yss/下载/1299_1798319416187011072.json'):
    new = threading.Thread(target=worker, name='new')
    new.start()
    time.sleep(3)
    url = f"http://10.0.3.13:5000/pointcloud/3dbox" 
    urls, frame_names = get_all_urls(json_file)
    print('urls: ', urls)
    print('frame_names: ', frame_names)
    for i in range(len(urls)):
        fordata = {'datas': [{'id': 0, 'pointCloudFile': urls[i]}], 'params': {}}
        res = requests.post(url, data=json.dumps(fordata), timeout=3)
        save_result_and_pointcloud(frame_names[i], res.text, urls[i])






if __name__ == '__main__':
    #predict_json_file()
    #exit(0  )

    new = threading.Thread(target=worker, name='new')
    new.start()
    time.sleep(3)

    # 试验外部调用加法函
    url = f"http://10.3.0.171:5000/pointcloud/3dbox"       # 服务端地址10.10.31.70   10.0.3.13
    #url = f"http://10.0.3.13:5000/pointcloud/3dbox"       # 服务端地址10.10.31.70   
    #pointCloudFile = 'https://test-cloud-control-1258234669.cos.ap-guangzhou.myqcloud.com/217/1263/2af4da02bf6b4e89afe568cca87cbe38/samples_new_2/point_cloud/1701757144425.pcd?sign=q-sign-algorithm%3Dsha1%26q-ak%3DAKIDkmg38ZRpKgNXOUQZxNwMqFYhBQ60FaSKjEjP3x3gJj6IuQZ96K3_71d2nqX84Y-G%26q-sign-time%3D1717725372%3B1717728972%26q-key-time%3D1717725372%3B1717728972%26q-header-list%3Dhost%26q-url-param-list%3D%26q-signature%3Dad3785e49e2c32e78fd8a5c03f57933184067d20&x-cos-security-token=ruiQcBQ4rQ02k91pNo28yXiKTwQLABIa93859b0fa5f97ec8736a469c6180ba4cWWtXMmwFAOg3LVAg3dh11-FIA2LpjiNy2rjwwl2PnB-WdjhglQUMYi4x0hrvmEGiGBfJtGr42eJ4Y65KqnqhQ28o59zFOCX3Y7sNK_QNDdylYUM_qF8ACfH1-Bz2i_TDU8Tq8sb4NbevJ6CJNZEW0LzybA52Hm_6WiqyICt8Hz1Xa1Sol64TudpYAZ3UfZx8EyVyvJWAmfups6CtllDe3ySUGX2d_MmYzCQMEIevFjuO0Olea0Cg0-gb6CHHiQkgpc2D3NQMLx6vd88MhcU_Fy2qvHLXzEWbZz2nZMbnBPYh_lDT9PzeX2MEnxioTpTsFo-Im6mFgfKiQ_lh-bV3vKTWjKIMYMLvAY7QZPK2lkvcDGtPEeUbZ-v0AJtHa2UxLWL7U4BU8QOgXyHC4TzqX4TXCkk1lyQy7fgCiUa6Eeb8c1-drr7keD89bJkwr_d2Q6bc6_dIr8K4Wp7euHpx23zC1j610SWn2jozsh7_VrDvkp11PSpy4LuFoAh-kV3dGSEkNBt41qwQsur0X_kC639RamOpBgmA5YmCE8Cbil0vG-SlSZOy4LvOg8xE0EUuKv60o5fYvKA4bx8VXsxrDKpj4zhYexO_kQ0zMag2_ywtR4W694PYcBkijf7v5vnzTYGL9MF8afE9zCJFlO0Zlf_E2RH0CdtO3aeW1jRQpTzlxRJ6_12iU_Aek-zeEcV7pvUAMoDht34CBIRSCgmjPCssqinwxO1GUUwEbbTr3WtNcOeMwL00Gyv4qcmwPe4f'
    pointCloudFile = 'https://test-cloud-control-1258234669.cos.ap-guangzhou.myqcloud.com/40/1145/11db035c22f54db1858d781cbbf9063e/GACRT-015_1715761798_ruqi2/point_cloud/1715761808300000000_640_102.pcd?sign=q-sign-algorithm%3Dsha1%26q-ak%3DAKIDbsBImJTKQTdIt5RG8QobovXFlnSCkNPuOolNx1Oy0KYjp8z8wbybAqBZzIGgnGF2%26q-sign-time%3D1718702559%3B1718832159%26q-key-time%3D1718702559%3B1718832159%26q-header-list%3Dhost%26q-url-param-list%3D%26q-signature%3D127f968dc6ecfb766ca639bbc73efbb2e3496516&x-cos-security-token=5yN5eMAXDoOTlW4cYA712xYaLe9A1IIa972c8946ba04a6e04957b5f535f35ff0Kc7fC-vCdgAqLNCBIZdTdYK1QGBgsFrOPWzf4Ar-YWXckYzIkQPNv-HIDY5CHfbNisL9NjhJOjmQsonA6lpXcEU7BFGnYSlNcWOc-ci8SXgdRAzdVMsnWl80QgisVBeCnCOkEL0blFklh_PsdW3oXP-tgXlU8T4veI348NN6RuKBNK7E63fmaY3gNqhqULfzj52F8Tb095uRtr2-qliU536aQ_FIuRqMejl8B0sbbNXX3_3tFWj54bcj2l8xGFJmlvSIwEP9WgLFUst1SxJk84lxex9WU6o5L8tpQG5suXvDsuOHkAeiNQAcwT-tHSFm1t7kd42SojmknKIsb2YB5pE7tAbEN1sq_0WyvL8cLmRoiC5BazzoG6cRMlaijGL3kBKTE2fy1mLTBleLTdLGEfNxMlkt8IPJW3aHEjpi2GRJk0V4PGb2HDaBW3yb98W-wg4LdxYNpFtSxx7EOMineXHA0psZYHnY2Y81k5Q_S6Wvp9jq1ap4DPhvI4VVtWoLdZCu_KgKJc_GfKrFX6epUmndBoVxf-JRLSkV_qrGAa_6pPmGqKHR1oNSQ7WHh1cq_eS5pLVRI5vx6N4ArUGTF5Gm2sIuNTSp5TdOSdtH9mcNxsA4LINUvY7OZLTXMkkRh8A1iO7IBDcPqFyNs60M_EykXwgO7AVc_weCcRyMgpRLU65whDOuvbpGmIFjicmwYer6mL1_IDgW6SV6a8BNXhk42br2FhHVVyEEc7w_fvPlV1UxYVdE-1qqRCAe_bkN'
    fordata = {'datas': [{'id': 0, 'is_encryption_pointcloud':1, 'pointCloudFile': pointCloudFile}], 'params': {}}     # 请求参数及目的地址
    #fordata = {'datas': [{'id': 0, 'pointCloudFile': pointCloudFile}], 'params': {}}     # 请求参数及目的地址
    
    for i in range(1000):
        res = requests.post(url, data=json.dumps(fordata), timeout=10)
        print(res)
        print(res.text)
        print("i: ", i)
    with open('data.json', 'w') as f:
        json.dump(json.loads(res.text), f , indent = 4)
    
