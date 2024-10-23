import json
import math
import quaternion
import numpy as np



def get_frame_position_local(path):
    position = []
    with open(path, 'r') as f:
        data = json.load(f)
        position.append(data["position"]["x"])
        position.append(data["position"]["y"])
        position.append(data["position"]["z"])
        position.append(data["position"]["roll"] * 3.14159 / 180)
        position.append(data["position"]["pitch"] * 3.14159 / 180)
        position.append(data["position"]["yaw"] * 3.14159 / 180)

        position[3] = -position[3]
        position[4] = -position[4]
        position[5] = 3.14159/2 - position[5]
    return position

def get_camera_config(path):
    world_para_all = []
    width_height_all = []
    inner_para_all = []
    with open(path, 'r') as f:
        data = json.load(f)
        for camera_conf in data:
            world_para = []
            width_height = []
            inner_para = []
            inner_para.append(float(camera_conf["camera_internal"]["fx"]))
            inner_para.append(float(camera_conf["camera_internal"]["fy"]))
            inner_para.append(float(camera_conf["camera_internal"]["cx"]))
            inner_para.append(float(camera_conf["camera_internal"]["cy"]))

            width_height.append(data[0]["width"])
            width_height.append(data[0]["height"])

            world_row1 = []
            world_row1.append(float(camera_conf["camera_external"][0]))
            world_row1.append(float(camera_conf["camera_external"][1]))
            world_row1.append(float(camera_conf["camera_external"][2]))
            world_row1.append(float(camera_conf["camera_external"][3]))
            world_para.append(world_row1)
            world_row1 = []
            world_row1.append(float(camera_conf["camera_external"][4]))
            world_row1.append(float(camera_conf["camera_external"][5]))
            world_row1.append(float(camera_conf["camera_external"][6]))
            world_row1.append(float(camera_conf["camera_external"][7]))
            world_para.append(world_row1)
            world_row1 = []
            world_row1.append(float(camera_conf["camera_external"][8]))
            world_row1.append(float(camera_conf["camera_external"][9]))
            world_row1.append(float(camera_conf["camera_external"][10]))
            world_row1.append(float(camera_conf["camera_external"][11]))
            world_para.append(world_row1)
            world_row1 = []
            world_row1.append(float(camera_conf["camera_external"][12]))
            world_row1.append(float(camera_conf["camera_external"][13]))
            world_row1.append(float(camera_conf["camera_external"][14]))
            world_row1.append(float(camera_conf["camera_external"][15]))
            world_para.append(world_row1)
            inner_para_all.append(inner_para)
            width_height_all.append(width_height)
            world_para_all.append(world_para)
    return inner_para_all, width_height_all, world_para_all
        
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
############################################################################
def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    quatern = np.quaternion(w,x,y,z)
    return quatern

def rotate_quaternion(point, quatern):
    point = quaternion.from_vector_part(point)
    p_new = quatern * point * quatern.conjugate()
    p_new = quaternion.as_vector_part(p_new)
    return p_new

def trans_point_world(point, position):
    quatern = euler_to_quaternion(position[3], position[4], position[5])
    p_rotated = rotate_quaternion(point, quatern)
    p_rotated[0] = p_rotated[0] + position[0]
    p_rotated[1] = p_rotated[1] + position[1]
    p_rotated[2] = p_rotated[2] + position[2]
    return p_rotated

def trans_label_world(label, position):
    label_out = label
    conners = get_conners(label)
    conner2 = []
    quatern = euler_to_quaternion(position[3], position[4], position[5])
    for point in conners:
        p_rotated = rotate_quaternion(point, quatern)
        p_rotated[0] = p_rotated[0] + position[0]
        p_rotated[1] = p_rotated[1] + position[1]
        p_rotated[2] = p_rotated[2] + position[2]
        conner2.append(p_rotated)

    for i in range(3):
        value_all = 0.
        for j in range(len(conner2)):
            value_all = value_all + conner2[j][i]
        value_all = value_all/float(len(conner2))
        label_out[i] = value_all
    
    length1 = math.sqrt((conner2[0][0] - conner2[1][0])*(conner2[0][0] - conner2[1][0]) + (conner2[0][1] - conner2[1][1])*(conner2[0][1] - conner2[1][1]))
    length2 = math.sqrt((conner2[2][0] - conner2[1][0])*(conner2[2][0] - conner2[1][0]) + (conner2[2][1] - conner2[1][1])*(conner2[2][1] - conner2[1][1]))
    if length1 > length2:
        angle = math.atan2(conner2[0][1] - conner2[1][1], conner2[0][0] - conner2[1][0])
        label_out[6] =  angle
    else:
        angle = math.atan2(conner2[2][1] - conner2[1][1], conner2[2][0] - conner2[1][0])
        label_out[6] =  angle
    return label_out

def word_point_car(point, position):
    p_car = point
    p_car[0] = p_car[0] + position[0]
    p_car[1] = p_car[1] + position[1]
    p_car[2] = p_car[2] + position[2]
    quatern = euler_to_quaternion(-position[3], -position[4], -position[5])
    p_rotated = rotate_quaternion(p_car, quatern)
    return p_rotated
###########################################################################
def cloud_to_image(point, inner_para, width_height, world_para):
    #inner_array = np.array(inner_para)
    inner_para_tep = [[inner_para[0], 0., inner_para[2], 0.],[0., inner_para[1], inner_para[3], 0.],[0., 0., 1., 0.]]
    inner_array = np.array(inner_para_tep)
    world_array = np.array(world_para)
    point_array = np.array([point[0], point[1], point[2], 1.])
    p_result = np.dot(world_array, point_array)
    if p_result[2] < 0:
        return -1, -1
    p_result_tep = np.dot(inner_array, p_result)
    u = int(p_result_tep[0] / p_result_tep[2])
    v = int(p_result_tep[1] / p_result_tep[2])
    if u >= 0 and u < width_height[0] and v >= 0 and v < width_height[1]:
        return u, v
    return -1, -1













