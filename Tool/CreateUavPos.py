import random
import numpy as np
from array_data_zhuanyi import *

def Is_enter_buildings(uav, buildings_location):
    x = uav[0]
    y = uav[1]
    z = uav[2]
    grid_x = int(x)
    grid_y = int(y)
    height = buildings_location[grid_x][grid_y]
    if z <= height:
        return True
    return False
def is_collision(uav_positions, x, y, radius):
    for pos in uav_positions:
        if np.linalg.norm([pos[0] - x, pos[1] - y]) < 2 * radius:
            return True
    return False


def generate_uav_pos(x1, x2, y1, y2):
    x = random.randint(x1, x2)
    y = random.randint(y1, y2)
    return x, y


def generate_uav_pos_array():
    uav_pos = [[] for _ in range(34)]
    uav_r = 0.3
    z = 0
    # Group 1: 8 UAVs with x in the range [0, 15] and y in the range [0, 15]
    for i in range(8):
        while True:
            x, y = generate_uav_pos(35, 49, 0, 49)
            if not is_collision(uav_pos[:i], x, y, uav_r):
                break
            if not Is_enter_buildings([x, y, 0], buildings_location_zhuanyi):
                break
        uav_pos[i].append(x)
        uav_pos[i].append(y)
        uav_pos[i].append(z)

    # Group 2: 9 UAVs with x in the range [35, 49] and y in the range [0, 15]
    for i in range(8, 17):
        while True:
            x, y = generate_uav_pos(0, 48, 0, 15)
            if not is_collision(uav_pos[:i], x, y, uav_r):
                break
            if not Is_enter_buildings([x, y, 0], buildings_location_zhuanyi):
                break
        uav_pos[i].append(x)
        uav_pos[i].append(y)
        uav_pos[i].append(z)

    # Group 3: 8 UAVs with x in the range [0, 15] and y in the range [35, 49]
    for i in range(17, 25):
        while True:
            x, y = generate_uav_pos(0, 35, 0, 49)
            if not is_collision(uav_pos[:i], x, y, uav_r):
                break
            if not Is_enter_buildings([x, y, 0], buildings_location_zhuanyi):
                break
        uav_pos[i].append(x)
        uav_pos[i].append(y)
        uav_pos[i].append(z)
    # Group 4: 9 UAVs with x in the range [35, 49] and y in the range [35, 49]
    for i in range(25, 34):
        while True:
            x, y = generate_uav_pos(0, 49, 35, 49)
            if not is_collision(uav_pos[:i], x, y, uav_r):
                break
            if not Is_enter_buildings([x, y, 0], buildings_location_zhuanyi):
                break
        uav_pos[i].append(x)
        uav_pos[i].append(y)
        uav_pos[i].append(z)

    return uav_pos

array = generate_uav_pos_array()
print(array)
