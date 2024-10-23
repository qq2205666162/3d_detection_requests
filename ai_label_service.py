import argparse
from flask import Flask, request
import requests
import json
from datetime import datetime
import threading
import asyncio
import aiohttp
import duckdb

from trans_axis import trans_label_world
import conf
import rsa_aes_util


app = Flask(__name__)



if 0:
    # asyncio request
    def reques(reqst):
        bg = datetime.now()
        print(f'{bg} start asyncio request')
        done, pending = asyncio.run(main(reqst))
        done = [e.result() for e in list(done)]
        print(f'{datetime.now()} finish reques, time spent {datetime.now() - bg}')
        print(f'{datetime.now()} pending fusion: {pending}')
        return done, pending
else:
    # normal request
    def reques(reqst):
        ret = []
        bga = datetime.now()
        for i, req in enumerate(reqst):
            bg = datetime.now()
            print(f"{bg} 发送第{i}个请求：", req)
            try:
                res = requests.post(req['url'], data=json.dumps(req['data']), timeout=config.timeout)

                req['status'] = res.status_code
                req['result'] = json.loads(res.text)
            except Exception as e:
                print(f'{datetime.now()} excetion: {e}')
                req['status'] = f'{e}'
                req['result'] = {}
            ret.append(req)
            print(f'{datetime.now()} received results, time spent {datetime.now() - bg}')
        print(f'{datetime.now()} finish reques, time spent {datetime.now() - bga}')
        return ret, []


async def fetch(session, req):
    print("发送请求：", req)
    response = await session.post(req['url'], data=json.dumps(req['data']))
    req['status'] = response.status
    try:
        req['result'] = json.loads(await response.text())
    except:
        req['result'] = await response.text()
    return req 


async def main(reqst):
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(fetch(session, r)) for r in reqst]
        res = await asyncio.wait(tasks)
    return res


@app.route('/AILabel/', methods=["GET", "POST"])
def inference():
    try:
        kwargs = json.loads(request.data)
        paras = kwargs['params']  # avoid the error that local variable 'paras' referenced before assignment when exceptions happened before paras is defined
        print(f'\n\n{datetime.now()} new request, received arguments:\n {kwargs}')

        for e in kwargs['datas']:
            if e['is_encryption_position'] == 0: 
                e['position'] = json.loads(requests.get(e['position'], allow_redirects=True).content)#######################################################################
            if e['is_encryption_position'] == 1:
                e['position'] = rsa_aes_util.getting_json(e['position'])
        
        if kwargs['is_encryption_position_camera_conf'] == 0:
            kwargs['params']['camera_conf'] = json.loads(requests.get(kwargs['params']['camera_conf'], allow_redirects=True).content)#######################################
        if kwargs['is_encryption_position_camera_conf'] == 1:
            kwargs['params']['camera_conf'] = rsa_aes_util.getting_json(kwargs['params']['camera_conf'])

        for ij, c in enumerate(kwargs['params']['camera_conf']):
            kwargs['params']['camera_conf'][ij] = {e: c[e] for e in ['camera_internal', 'width', 'height', 'camera_external', 'rowMajor']}


        print(f'{datetime.now()} get position and camera configuration done !')

        secName = kwargs['params'].get('secName', datetime.now().strftime("%Y-%m-%d %H-%M-%S"))

        qr = f'select unnest({kwargs}, recursive:=true)'  # ['datas', 'secName'(optional), 'camera_conf'], (1, 2)
        qr = f'select unnest(datas, recursive:=true) from ({qr})'  # ['id', 'frameName'(optional), 'images', 'pointCloudFile', 'name', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'type']
        qr = f'select * exclude(id), id as frameID from ({qr})'

        ############################################################## image & pcd object detection  #############################################################
        print(f'{datetime.now()} start object detection...')
        reqst = []
        paras = kwargs['params']
        # image and pcd object detection
        for i, e in enumerate(kwargs['datas']):######################################################################################
            reqst.append({'data': {'datas': [{'id': e['id'],"is_encryption_pointcloud": e['is_encryption_pointcloud'], 'pointCloudFile': e['pointCloudFile']}], 'params': {}}, 'url': config.pcd3dBox, 'frameID': e['id'], 'frameName': e.get('frameName', ''), 'cameraID': -1, 'dtype': 'pcd'})  # only pass the necessary arguments
            for k, v in enumerate(e['images']):
                reqst.append({'data': {"model_use": 0, "is_encryption_image": e['is_encryption_image'][k], "img_url": v}, "url": config.image2dBox, 'frameID': e['id'], 'frameName': e.get('frameName', ''), 'cameraID': k, 'dtype': 'image'})
        print(f'{datetime.now()} len(reqst): {len(reqst)}')
        
        if config.debug:
            pickle.dump(reqst, open(f'detection_request {secName}.pkl', 'wb'))

        done, pending = reques(reqst)
        print(f'{datetime.now()} pending detection: {pending}')
        print(f'{datetime.now()} len(done) detection: {len(done)}')

        excepts = {}
        excepts['pcd3d'] = [e for e in done if e['dtype']=='pcd' and (e['result']=={} or e['result']['data'][0]['message']!='success')]
        excepts['image2d'] =  [e for e in done if e['dtype']=='image' and e['result'].get('message', None)!='success']

        if config.debug:
            pickle.dump(done, open(f'detection_done {secName}.pkl', 'wb'))
        qd = f''' select unnest({[{'frameID': e['frameID'], 'result': e['result']} for e in done if e['dtype']=='pcd' and e['result']!={} and e['result']['data'][0]['message']=='success']}, recursive:=true) '''  # ['frameID', 'code', 'message', 'data']
        qd = f''' select frameID, unnest(data, recursive:=true) from ({qd}) '''  # ['frameID', 'id', 'code', 'message', 'classes']
        qd = f''' select *, len(classes) as nbox, [e['confidence'] for e in classes] as confidence from ({qd})'''

        ################################################################ image & pcd cross-check  #############################################################
        print(f'{datetime.now()} start cross-check...')
        done.sort(key=lambda k: (k['frameID'], k['cameraID']), reverse=False)
        num_camera = len(kwargs['datas'][0]['images'])
        reqst = []

        inner_para = [[paras['camera_conf'][j]['camera_internal'][e] for e in ['fx', 'fy', 'cx', 'cy']] for j in range(num_camera)]  # [[fx, fy, cx, cy], ], 相机-相机内参，把点云3D框转到2D框
        width_height = [[paras['camera_conf'][j][e] for e in ['width', 'height']] for j in range(num_camera)]  # [[width, height], ]
        world_para = [[paras['camera_conf'][j]['camera_external'][k*4: (k+1)*4] for k in range(4)] for j in range(num_camera)]  # [[旋转平移矩阵16个数, 4*4]], 相机-相机外参， 把点云3D框转到图像坐标系, 再通过inner_para把图像坐标系转成2D框

        for i in range(int(len(done)/(num_camera+1))):
            reqst.append({'url': config.fusion, 
                          'frameID': done[i*(num_camera+1)]['frameID'],
                'data':{
                "labels": [[e['x'], e['y'], e['z'], e['dx'], e['dy'], e['dz'], e['rotZ']] for e in done[i*(num_camera+1)]['result'].get('data', [{}])[0].get('classes', [])],  # [[x, y, z, dx, dy, dz, yaw], ], 雷达坐标系下的，原3D检测服务输出结果
                "image_boxes": [[[e['box']['x1'], e['box']['y1'], e['box']['x2'], e['box']['y2']] for e in done[i*(num_camera+1)+j+1]['result'].get('data', [])] for j in range(num_camera)],  # e.g. [[[863.7344970703125, 966.3215942382812, 1435.7830810546875, 1200.177978515625]],  ], 相机-框-[x1,y1,x2,y2]
                "inner_para": inner_para,
                "width_height": width_height,
                "world_para": world_para,
                "destination_adadressdress":"  http://10.10.17.23:26001/pvprocess/  "  # 调用方地址
                }})
        print(f'{datetime.now()} len(reqst): {len(reqst)}')
        
        if config.debug:
            pickle.dump(reqst, open(f'fusion_request {secName}.pkl', 'wb'))

        done, pending = reques(reqst)
        print(f'{datetime.now()} pending fusion: {pending}')
        print(f'{datetime.now()} len(done) fusion: {len(done)}')

        if config.debug:
            print(f'{datetime.now()} save fusion_done')
            pickle.dump(done, open(f'fusion_done {secName}.pkl', 'wb'))
        
        print(f'{datetime.now()} start re-construct result')
        for e in done:
            if e['result'] == {}:
                e['status'] = 200
                e['result'] = {'result': [0.] * len(e['data']['labels'])} 
        qf = f''' select unnest({done}, recursive := true) '''  # ['url', 'frameID', 'labels', 'image_boxes', 'inner_para', 'width_height', 'world_para', 'destination_adadressdress', 'status', 'result']
        qf = f''' select frameID, len(labels) as nbox, result::float[] as result from ({qf}) '''


        ################################################## transform to world coordinates  #############################################################
        print(f'{datetime.now()} start transform to world coordinates ')

        con = duckdb.connect()
        
        # if some (frameID, nbox) is missing from qf, the corresponding column 'boxes' and 'nboxNew' in the resulting qdf will be NULL
        qdf = f''' select frameID, list_where(classes, list_transform(result, (x, xi) -> (x + confidence[xi])>{config.cross_check_threshold})) as boxes, nbox, len(boxes) as nboxNew from ({qd}) left join ({qf}) using(frameID, nbox) '''  # ['frameID', 'boxes', 'nbox', 'nboxNew']
        
        if 1:
            print(f'{datetime.now()} start create_function ')
            
            con.create_function("trans_label_world", trans_label_world, ['float[]', 'float[]'], 'float[]')
            qrdf = f'select * from ({qr}) left join ({qdf}) using(frameID)'  # ['frameName', 'images', 'pointCloudFile', 'name', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'type', 'frameID', 'boxes', 'nbox', 'nboxNew']
            qrdf = f''' select COLUMNS('(frameID|frameName)'), [x, y, z, -roll*pi()/180, -pitch*pi()/180, pi()/2 - yaw*pi()/180] as pos, unnest(boxes, recursive:=true) from ({qrdf})'''  # ['frameID', 'pos', 'label', 'confidence', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'rotX', 'rotY', 'rotZ']  # NOTE tansform angle 
            # trans_label_world(label, position), e.g. label=[x, y, z, dx, dy, dz, yaw] in lidar cooridnate system, position=[x, y, z, roll, pitch, yaw], return [x, y, z, dx, dy, dz, yaw]
            qrdf = f''' select *, trans_label_world([x, y, z, dx, dy, dz, rotZ]::float[], pos::float[]) as box_transformed from ({qrdf})  '''

            print(f'{datetime.now()} start create frames ')

            frames = con.sql(qrdf).project(f'*, row_number() OVER () as rowid').aggregate('''COLUMNS('(frameID|frameName)'), list({'x': x::float, 'y': y::float, 'z': z::float, 'dx': dx::float, 'dy': dy::float, 'dz': dz::float, 'rotX': rotX::float, 'rotY': rotY::float, 'rotZ': rotZ::float, 'label': label, 'confidence': confidence::float} order by rowid) as boxes, list(label order by rowid) as labels, list(box_transformed order by rowid) as boxes_transformed ''').project('row_number() over(order by frameID)-1 as idx, *')  # ['idx', 'frameID', 'frameName', 'boxes', 'labels', 'boxes_transformed']  # The frameID passed from the requests may not start from 0, using a map to associate frameID with frameIdx returned from tracking
            if 1:
                print(f'{datetime.now()} start re-create frames ')

                con.sql(f'''create or replace temp table frames as select row_number() over(order by frameID)-1 as idx, * exclude(idx) from (select frameID, frameName from ({qr})) left join frames using(frameID, frameName)''')
                con.sql('update frames set boxes=[], labels=[], boxes_transformed=[] where boxes is null')
                frames = con.table('frames')
        else:
            frames = con.sql(qdf).project(''' frameID, boxes::STRUCT("label" VARCHAR, confidence float, x float, y float, z float, dx float, dy float, dz float, rotX float, rotY float, rotZ float)[] as boxes, [e['label'] for e in boxes] as labels, [[e['x'], e['y'], e['z'], e['dx'], e['dy'], e['dz'], e['rotZ']] for e in boxes]::float[][] as boxes_transformed ''')

        ################################################################ tracking  #############################################################
        print(f'\n{datetime.now()} start tracking...')

        track_input = frames.aggregate('list(labels order by frameID) as labels, list(boxes_transformed order by frameID) as boxes').fetchall()[0]
        track_input = {'data_json': track_input[1], 'cat_json': track_input[0], "destination_adadressdress": "http://10.10.17.23:26001/pvprocess/"}

        if config.debug:
            print(f'{datetime.now()} save track_reqst')
            pickle.dump(track_input, open(f'track_reqst {secName}.pkl', 'wb'))

        print(f'{datetime.now()} start json.dumps(track_input) ')

        data = json.dumps(track_input, indent=4)
        # print(f'{datetime.now()} send track request:\n{data}')
        tmp = datetime.now()
        res = requests.post(config.track, data=data, timeout=120)
        print(f'{datetime.now()} finish reques, time spent {datetime.now()-tmp}, result:\n{res}')

        ################################################################ combine results  #############################################################
        print(f'{datetime.now()} start combine results ')

        q = f''' select unnest({json.loads(res.text)['result']}) as obj '''
        q = f''' select *, row_number() over()-1 as objID from ({q}) '''
        q = f''' select objID, unnest(obj) as obj from ({q})  '''
        q = f''' select objID, obj[1] as idx, obj[2] as boxID from ({q})  '''
        print(f'{datetime.now()} start create tracked ')

        tracked = con.sql(q)
        
        q = f''' select idx, COLUMNS('(frameID|frameName)'), unnest(boxes) as box, unnest(range(len(boxes))) as boxID from frames ''' 
        q = f''' select * from tracked l join ({q}) r using(idx, boxID) '''  # ['objID', 'idx', 'boxID', 'frameID', 'frameName', 'box']
        q = f''' select COLUMNS('(frameID|frameName)'), list(struct_insert(box, objID := objID||'_'||strftime(now(), '%Y%m%d%H%M%S%f')) order by boxID) as boxes from ({q}) group by all'''  # ['frameID', 'frameName', 'boxes']

        print(f'{datetime.now()} start create rest ')

        rest = con.sql(q).project('frameID as id, * exclude(frameID)')
        rest = con.sql('select rest from rest').aggregate(''' list(rest order by rest.id) as frame_boxes ''').project(''' {'code': 0, 'message': '', 'frame_boxes': frame_boxes} ''').fetchall()[0][0]
        rest['secName'] = paras.get('secName', '')
        rest['exceptions'] = excepts
        
    except Exception as e:
        print(e)
        rest = {
            "code": 1,
            "message": {'Binder Error: UNNEST not supported here': 'dependent service failed'}.get(f'{e}', f'{e}'),
            'secName': paras.get('secName', '')
        }

    if config.debug:
        print(f'{datetime.now()} save rest ')
        pickle.dump(rest, open(f'rest {secName}.pkl', 'wb'))

    print(f'{datetime.now()} finish section {secName}')
    return json.dumps(rest)

def worker(host, port):
    # 启动端口监控
    app.run(host=host, port=port)


def parse_args():
    parser = argparse.ArgumentParser(description='3D detection and tracking')
    parser.add_argument('--env', type=str, default='production', help="environment to use, set to 'test' or 'production' ")
    parser.add_argument('--debug', action='store_true', default=True, help='using debug mode')
    parser.add_argument('--timeout', type=int, default=10, help="timeout when call image and point cloud detection service ")
    parser.add_argument('--cross_check_threshold', type=float, default=0.15, help="image and point cloud cross check threshold ")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    if 0:
        import pickle

        reqst = pickle.load(open('D:/code/ailabel/fusion', 'rb'))
        len(reqst)
        done, pending = asyncio.run(main(reqst[:1]))
        print(done)

    args = parse_args()
    config = conf.config(ev=args.env, debug=args.debug, timeout=args.timeout, cross_check_threshold=args.cross_check_threshold)

    if config.debug:
        import pickle
        # from toolkit import Logger
        # log = Logger(fname=f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt', path="D:/code")

    host = "0.0.0.0"                 # 服务端IP地址
    port = 333                             # 服务端监听端口号
    new = threading.Thread(target=worker, args=(host, port), name='test')
    new.start()