#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import requests
import threading
from flask import Flask,request
from InterfaceAPI import *

accept_data = None
app = Flask(__name__)

# 监听,用于接收返回值
@app.route("/pvprocess/", methods=["GET", "POST"])
def data_accept():
    global accept_data
    comment = json.loads(request.data)
    accept_data = comment
    # print('posResult: ', accept_data)
    state = 200 if comment else 400
    return json.dumps({"err_code": state})

def worker():
    app.run(host='10.80.40.52',port=26001)      # host表示本机地址，port表示返回参数的监听端口

if __name__ == '__main__':

    new = threading.Thread(target=worker, name='new')
    new.start()
    time.sleep(3)

    # 试验外部调用加法函数
    url = f'http://10.80.40.52:33339/yolov8/'     # 服务端地址10.0.3.13
    url = f'http://10.0.3.13:33339/yolov8/' 
    
    adress = 'https://test-cloud-control-1258234669.cos.ap-guangzhou.myqcloud.com/40/1145/11db035c22f54db1858d781cbbf9063e/GACRT-015_1715761798_ruqi2/point_cloud/1715761808300000000_640_102.pcd?sign=q-sign-algorithm%3Dsha1%26q-ak%3DAKIDbsBImJTKQTdIt5RG8QobovXFlnSCkNPuOolNx1Oy0KYjp8z8wbybAqBZzIGgnGF2%26q-sign-time%3D1718702559%3B1718832159%26q-key-time%3D1718702559%3B1718832159%26q-header-list%3Dhost%26q-url-param-list%3D%26q-signature%3D127f968dc6ecfb766ca639bbc73efbb2e3496516&x-cos-security-token=5yN5eMAXDoOTlW4cYA712xYaLe9A1IIa972c8946ba04a6e04957b5f535f35ff0Kc7fC-vCdgAqLNCBIZdTdYK1QGBgsFrOPWzf4Ar-YWXckYzIkQPNv-HIDY5CHfbNisL9NjhJOjmQsonA6lpXcEU7BFGnYSlNcWOc-ci8SXgdRAzdVMsnWl80QgisVBeCnCOkEL0blFklh_PsdW3oXP-tgXlU8T4veI348NN6RuKBNK7E63fmaY3gNqhqULfzj52F8Tb095uRtr2-qliU536aQ_FIuRqMejl8B0sbbNXX3_3tFWj54bcj2l8xGFJmlvSIwEP9WgLFUst1SxJk84lxex9WU6o5L8tpQG5suXvDsuOHkAeiNQAcwT-tHSFm1t7kd42SojmknKIsb2YB5pE7tAbEN1sq_0WyvL8cLmRoiC5BazzoG6cRMlaijGL3kBKTE2fy1mLTBleLTdLGEfNxMlkt8IPJW3aHEjpi2GRJk0V4PGb2HDaBW3yb98W-wg4LdxYNpFtSxx7EOMineXHA0psZYHnY2Y81k5Q_S6Wvp9jq1ap4DPhvI4VVtWoLdZCu_KgKJc_GfKrFX6epUmndBoVxf-JRLSkV_qrGAa_6pPmGqKHR1oNSQ7WHh1cq_eS5pLVRI5vx6N4ArUGTF5Gm2sIuNTSp5TdOSdtH9mcNxsA4LINUvY7OZLTXMkkRh8A1iO7IBDcPqFyNs60M_EykXwgO7AVc_weCcRyMgpRLU65whDOuvbpGmIFjicmwYer6mL1_IDgW6SV6a8BNXhk42br2FhHVVyEEc7w_fvPlV1UxYVdE-1qqRCAe_bkN'
    #adress = 'http://localhost:24/20240112-093530.jpg'
    fordata = {"model_use": 0,"is_encryption_image": 1, "img_url":adress, "destination_address":"http://10.10.17.23:26001/pvprocess/"}     # 请求参数及目的地址
    
    # 试验外部调用阶乘函数
    #url = f'http://10.10.17.23:33339/factorial/'     # 服务端地址
    #fordata = {"a":10,"destination_address":"http://10.10.17.23:26000/pvprocess/"}          # 请求参数及目的地址
    
    print(f"json.dumps(fordata) = {json.dumps(fordata)}")
    postApi = InterfaceAPI(url, json.dumps(fordata))

    start_time = time.time()
    for i in range(1):
        postApi = InterfaceAPI(url, json.dumps(fordata))
        response = postApi.response
        print("i: ", i)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("程序运行耗时：", elapsed_time)

    cnt = 0
    response = postApi.response
    with open('result.json', 'w') as json_file:
        json_file.write(response.text)
    print(f"response = {response.text}")

    while response == -1:
        time.sleep(2)
        cnt += 1
        response = postApi.post()
        print('parms:', response, 'cnt: ', cnt)
        if cnt > 3:
            print('Algorithm not run!')
            break