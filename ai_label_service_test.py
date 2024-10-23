import requests
import json
from datetime import datetime
import argparse
from flask import Flask, request
import requests
import json
from datetime import datetime
import threading
import asyncio
import aiohttp
import duckdb
import time

from trans_axis import trans_label_world
import conf
import rsa_aes_util




def req(sec, fs, num_camera=7, position_dir='pos_config', camera_conf='new2_calibration_car2camera.json'):
    reqst = {
        "datas": [
            {
                "id": k,
                'frameName': f,
                "images": [f'http://dk8&nhL:38)nm%G&aoB0)qC*7(@10.30.22.14:47551/{sec}/image{i}/{f}.jpg' for i in range(num_camera)],
                "pointCloudFile": f"http://dk8&nhL:38)nm%G&aoB0)qC*7(@10.30.22.14:47551/{sec}/point_cloud/{f}.pcd",
                "position": f'http://dk8&nhL:38)nm%G&aoB0)qC*7(@10.30.22.14:47551/{sec}/{position_dir}/{f}.json',  # dictionaries, e.g. {"name": "L401_All_lidar_urbanroad_frame_50_1697444681863000_1577844830295423.pcd", "position": {"x": 742843.6280302228, "y": 2556638.846987676, "z": 4.263, "roll": 1.571044921875, "pitch": 0.439453125, "yaw": -93.5595703125}}
                "type": "SINGLE_DATA"
            }
        for k, f in enumerate(fs)],
        "params": {
            'secName': sec.split('/')[-1],
            'camera_conf': f'http://dk8&nhL:38)nm%G&aoB0)qC*7(@10.30.22.14:47551/{camera_conf}'  # list of dictionaries, each dictionary is like: {'camera_internal': {'fx': 1714.2696469129685, 'fy': 1714.2403097621539, 'cx': 1935.5921396452666, 'cy': 1035.2083615084966}, 'width': 3840, 'height': 2160, 'camera_external': [0.792771, -0.609431, 0.0103707, -1.25781, 0.010258, -0.00367222, -0.999941, 1.00304, 0.609432, 0.792831, 0.00334016, -2.16132, 0.0, 0.0, 0.0, 1.0], 'rowMajor': False}
        }
    }
    if 1:
        import json
        reqst = json.load(open('/home/yss/下载/1172_1803002872912465920.json', 'r'))
    reqst = json.dumps(reqst)

    bg = datetime.now()
    print(f'{bg} send request\n{reqst}')

    if 1:
        url = "http://127.0.0.1:333/AILabel/"
        url = "http://10.3.0.171:333/AILabel/"
        for xxxx in range(1000):
            r = requests.post(url, data=reqst).text
            print("loop : ", xxxx)
            print(f'{datetime.now()} returned text: {r}')
            #time.sleep(3)
            
    else:
        from ai_label_service import inference

        class A: pass
        a = A()
        a.data = reqst
        r = inference(a)

    # print(f'{datetime.now()} returned text: {r}')
    r = json.loads(r)
    print(f'{datetime.now()} received response: {r["code"]}, message: {r["message"]}, time spent {datetime.now() - bg}')
    return r


if __name__ == '__main__':
    if 0:
        r = req(sec='2023_1016_1_0_L401_urbanroad_frame_50_1697444679363000_1577844827800000', fs=['L401_urbanroad_frame_100_1697444681863000_1577844830300000', 
                    'L401_urbanroad_frame_110_1697444682363000_1577844830799999',
                    'L401_urbanroad_frame_120_1697444682863000_1577844831300000', 
                    'L401_urbanroad_frame_130_1697444683363000_1577844831800000',
                    'L401_urbanroad_frame_140_1697444683863000_1577844832300000',
                    ])
        print(r)

    if 1:
        r = req(sec='zhuxingxing/dataset/20231126_LNABLAB39N5506120-10_object_1-20240425165854', fs=['1700967571025', 
                    '1700967571525',
                    '1700967572025', 
                    '1700967572525',
                    '1700967573025',
                    ], num_camera=8, position_dir='position', camera_conf='zhuxingxing/dataset/20231126_LNABLAB39N5506120-10_object_1-20240425165854/camera_config/1700967590525.json')
        print(r)

    if 0:
        import os
        import duckdb

        con = duckdb.connect()
        gt = os.listdir('./ailabel/labels_down')
        len(gt)
        gt[:3]

        q = '''select column1 as sec, column2[:-5] as f from './ailabel/pcds.csv' '''
        q = f''' select *, split_part(f, '_', -1) as dtime from (select unnest({gt})[:-5] as f) join ({q}) using(f)'''
        tmp = con.sql(q)
        tmp.shape
        tmp

        q = f''' select sec, list(f order by dtime) as secf from ({q}) group by all order by sec'''
        tmp = con.sql(q)
        tmp
        tmp.fetchall()



        # from  ai_label_service import req

        bga = datetime.now()
        for i, e in enumerate(tmp.fetchall()):
            bg = datetime.now()
            print(f'{bg} process {i} secition: {e[0]}')
            print(f'number of frames {len(e[1])}')
            r = req(sec=e[0], fs=e[1])
            print(f'{datetime.now()} finish section, time spent: {datetime.now()-bg}')
            
        print(f'{datetime.now()} finish all, time spent: {datetime.now()-bga}')