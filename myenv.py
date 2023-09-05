import csv
import random
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
from array_data import *
from array_data_50 import *
from array_data_100 import *
from array_data_zhuanyi import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from array_data_WH import *
from WH_test import *
from matplotlib.image import imread
import matplotlib.style as mplstyle
from zhuanyi_test import *

"""
增加无人机数量、在空中形成五环形状、要避免无人机发生碰撞
"""
# 设置全局小数保留位数为2
np.set_printoptions(precision=2)
mplstyle.use('fast')


def is_arrive(uav_pos, aim_pos, tolerance=0.1):
    x_error = abs(uav_pos[0] - aim_pos[0])
    y_error = abs(uav_pos[1] - aim_pos[1])
    z_error = abs(uav_pos[2] - aim_pos[2])

    return x_error < tolerance and y_error < tolerance and z_error < tolerance


def is_arrive2D(uav_pos, aim_pos, tolerance=0.2):
    x_error = abs(uav_pos[0] - aim_pos[0])
    y_error = abs(uav_pos[1] - aim_pos[1])

    return x_error < tolerance and y_error < tolerance


def go_to(pointa, pointb, max_speed=0.3):
    x_diff = pointb[0] - pointa[0]
    y_diff = pointb[1] - pointa[1]

    # Calculate the distance to the target point
    distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

    # Check if x_diff or y_diff is less than 0.1 and set vx and vy accordingly
    if abs(x_diff) < 0.1:
        vx = 0
    else:
        vx_normalized = x_diff / distance
        vx = vx_normalized * max_speed

    if abs(y_diff) < 0.1:
        vy = 0
    else:
        vy_normalized = y_diff / distance
        vy = vy_normalized * max_speed

    return vx, vy


def is_collision(uav_positions, x, y, radius):
    for pos in uav_positions:
        if np.linalg.norm([pos[0] - x, pos[1] - y]) < 2 * radius:
            return True
    return False


def _will_enter_building(state, action, buildings_location, uav_r):
    next_x = state[0] + action[0]
    next_y = state[1] + action[1]
    next_z = state[2] + action[2]

    # 创建无人机的碰撞体积几何对象（球形）
    uav_collision_volume = Point(next_x, next_y, next_z).buffer(uav_r)

    # 获取无人机下一步的网格的位置
    grid_x = int(next_x)
    grid_y = int(next_y)

    """building_polygon = buildings_location[grid_x][grid_y]
    if isinstance(building_polygon, Polygon):
        if building_polygon.intersects(uav_collision_volume):
            return True
    """
    return False


def _is_in_building(state, buildings_location, uav_r):
    x = state[0]
    y = state[1]
    z = state[2]

    # 获取无人机当前的格子位置
    grid_x = int(x)
    grid_y = int(y)
    building_height = buildings_location[grid_x][grid_y]
    if z + uav_r <= building_height:
        return True

    return False


def _is_outside_map(state, action, map_w, map_h, map_z):
    next_x = state[0] + action[0]
    next_y = state[1] + action[1]
    next_z = state[2] + action[2]

    if next_x < 0 or next_x >= map_w or next_y < 0 or next_y >= map_h or next_z < 0 or next_z >= map_z:
        return True

    return False


def get_points_x(i):
    return points[i][0]


def get_points_y(i):
    return points[i][1]


def get_points(i):
    return points[i]


class UAVEnv(gym.Env):
    def __init__(self, uav_num, buildings, buildings_location, map_w, map_h, map_z, uav_r):
        super(UAVEnv, self).__init__()

        self.uav_num = uav_num
        self.buildings = buildings
        self.buildings_location = buildings_location
        self.map_w = map_w
        self.map_h = map_h
        self.map_z = map_z
        self.uav_r = uav_r
        self.position_pool = [[] for _ in range(self.uav_num)]

        # Define action and observation space  动作空间是所有动作状态的最大最小值之间、观察空间是所有状态的最大最小值之间
        self.action_space = spaces.Box(low=np.array([-0.3, -0.3, -0.3, 0] * self.uav_num),
                                       high=np.array([0.3, 0.3, 0.3, 1] * self.uav_num), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -1, -1, -1, 0] * self.uav_num),
                                            high=np.array([self.map_w, self.map_h, self.map_z, 1, 1, 1, 1] *
                                                          self.uav_num), dtype=np.float32)

        # Init state
        #self.state = uav_init_state
        self.state = [[0, 0, 0, 0, 0, 0, 0] for _ in range(32)]
        for i in range(32):
            x, y = match_pairs_zhuanyi[i][1][:2]
            self.state[i][:2] = x, y
        print(self.state)
        #self.state = uav_init_state_zhuanyi

    def recorder(self, env_t):
        for i in range(self.uav_num):
            x, y, z = self.state[i][:3]
            position = [x, y, z, env_t]
            self.position_pool[i].append(position)

    def step(self, actions, env_t):
        actions = np.array(actions).reshape(self.uav_num, 4)

        for i in range(self.uav_num):
            if _will_enter_building(self.state[i], actions[i], self.buildings_location, self.uav_r):
                actions[i][:3] = 0  # set vx, vy, vz to zero

            if _is_outside_map(self.state[i], actions[i], self.map_w, self.map_h, self.map_z):
                actions[i][:3] = 0  # set vx, vy, vz to zero

            # update state x，y，z位置更新为原来的加上偏移量；vx，vy，vz更新，
            self.state[i][0] += actions[i][0]  # uav_x = vx*t, suppose t=1
            self.state[i][1] += actions[i][1]  # uav_y = vy*t
            self.state[i][2] += actions[i][2]  # uav_z = vz*t
            self.state[i][3:6] = actions[i][:3]  # update vx, vy, vz
            self.state[i][6] = actions[i][3]  # update sensor status
        """
                    if _is_in_building(self.state[i], self.buildings_location, self.uav_r):
                self.state[i][0] -= actions[i][0]  # uav_x = vx*t, suppose t=1
                self.state[i][1] -= actions[i][1]  # uav_y = vy*t
                self.state[i][2] -= actions[i][2]  # uav_z = vz*t
        """

        return self.state, 0, False, {}

    def reset(self):
        self.state = np.zeros((self.uav_num, 7), dtype=np.float32)
        return self.state


class Render:
    def __init__(self, uav_num, state, buildings, map_w, map_h, map_z, uav_r, position_pool, map_size=10):
        self.uav_num = uav_num
        self.state = state
        self.buildings = buildings
        self.map_w = map_w
        self.map_h = map_h
        self.map_z = map_z
        self.uav_r = uav_r
        self.position_pool = position_pool
        self.map_size = map_size

        # 创建画布
        self.fig = plt.figure(figsize=(self.map_w, self.map_h))  # 设置画布大小
        self.ax = self.fig.add_subplot(111, projection='3d')  # 创建三维坐标系

        # 绘制网格
        for x in range(map_w + 1):
            self.ax.plot([x, x], [0, map_h], [0, 0], color='gray', alpha=0.5)
        for y in range(map_h + 1):
            self.ax.plot([0, map_w], [y, y], [0, 0], color='gray', alpha=0.5)

        # 绘制建筑
        # draw building

        for building in self.buildings:
            x = [building[0][0], building[1][0], building[3][0], building[2][0]]
            y = [building[0][1], building[1][1], building[3][1], building[2][1]]
            z = [building[0][2], building[1][2], building[3][2], building[2][2]]
            building_type = building[4]

            if building_type == 0:
                continue

            if building_type == 1:
                height = 1
                color = 'green'
            elif building_type == 2:
                height = 2
                color = 'orange'
            elif building_type == 3:
                height = 3
                color = 'purple'

            # Draw solid cuboid
            vertices = [
                [x[0], y[0], z[0]],
                [x[1], y[1], z[1]],
                [x[2], y[2], z[2]],
                [x[3], y[3], z[3]],
                [x[0], y[0], z[0] + height],
                [x[1], y[1], z[1] + height],
                [x[2], y[2], z[2] + height],
                [x[3], y[3], z[3] + height]
            ]
            faces = [
                # [0, 1, 2, 3],  # bottom face 不打印底面减小性能开销
                [4, 5, 6, 7],  # top face
                [0, 1, 5, 4],  # front face
                [1, 2, 6, 5],  # right face
                [2, 3, 7, 6],  # back face
                [3, 0, 4, 7]  # left face
            ]

            cuboid = Poly3DCollection([[vertices[point] for point in face] for face in faces], facecolors=color,
                                      linewidths=1, edgecolors='black', alpha=1)
            self.ax.add_collection3d(cuboid)

        self.ax.set_xlim(0, map_w + 1)
        self.ax.set_ylim(0, map_h + 1)
        self.ax.set_zlim(0, map_z + 1)

    def render3D(self):
        plt.ion()
        # draw uav
        for i in range(self.uav_num):
            x_traj, y_traj, z_traj, _ = zip(*self.position_pool[i])
            self.ax.plot(x_traj, y_traj, z_traj, color='gray', alpha=0.7, linewidth=2.0)


class CreateMap:
    def __init__(self, map_size, map_w, map_h, map_z):
        self.buildings = []  # 记录建筑四点位置和高度
        self.buildings_location = []  # 记录建筑中心位置和高度
        self.map_size = map_size
        self.map_w = map_w
        self.map_h = map_h
        self.map_z = map_z

    def get_Buildings(self):
        if self.map_size == 10:
            self.buildings = buildings_10
            self.buildings_location = buildings_location_10

        elif self.map_size == 50:
            self.buildings = buildings_zhuanyi
            self.buildings_location = buildings_location_zhuanyi

        elif self.map_size == 100:
            self.buildings = buildings_100
            self.buildings_location = buildings_location_100

        return self.buildings, self.buildings_location


def get_whz(size=10):
    map_w, map_h, map_z = 0, 0, 0
    if size == 10:
        map_w, map_h, map_z = 10, 10, 10
    elif size == 50:
        map_w, map_h, map_z = 50, 50, 10
    elif size == 100:
        map_w, map_h, map_z = 100, 100, 10
    return map_w, map_h, map_z


def main():
    uav_num = 32
    uav_r = 0.3
    map_size = 50
    env_t = 0
    map_w, map_h, map_z = get_whz(map_size)
    # 初始化MAP模块
    MAP = CreateMap(map_size, map_w, map_h, map_z)
    # 调用get_random_Buildings方法生成建筑物列表
    buildings, buildings_location = MAP.get_Buildings()
    # 初始化Env模块
    env = UAVEnv(uav_num, buildings, buildings_location, map_w, map_h, map_z, uav_r)
    # 初始化render模块
    render = Render(uav_num, env.state, buildings, map_w, map_h, map_z, uav_r, env.position_pool, map_size)

    actions = [[0, 0, 0, 0] for _ in range(32)]
    while env_t < 25:
        actions = [[0, 0, 0.2, 0] for _ in range(32)]
        obs, reward, done, info = env.step(actions, env_t)
        env.recorder(env_t)
        render.render3D()
        plt.pause(0.001)
        env_t += 1
    actions = [[0, 0, 0, 0] for _ in range(32)]
    while env_t < 30:
        for i in range(16, 32):
            actions[i] = [0, 0, 0.2, 0]
        obs, reward, done, info = env.step(actions, env_t)
        env.recorder(env_t)
        render.render3D()
        plt.pause(0.001)
        env_t += 1
    actions = [[0, 0, 0, 0] for _ in range(32)]
    done = False
    while not done:
        for match_pair in match_pairs_zhuanyi:
            index = match_pair[0]
            uav = match_pair[1]
            aim = match_pair[2]
            vx, vy = go_to(uav, aim)
            actions[index] = [vx, vy, 0, 0]
            if is_arrive2D(env.state[index][:2], aim):
                actions[index] = [0, 0, -0.2, 0]
                if env.state[index][2] < 0.2:
                    actions[index] = [0, 0, 0, 0]
                    print(f'{index} is ok')
                    render.ax.scatter(env.state[index][0], env.state[index][1], env.state[index][2], color='red', s=50)
        obs, reward, done, info = env.step(actions, env_t)
        # 循环画图
        env.recorder(env_t)
        render.render3D()
        plt.pause(0.001)
        env_t += 1
        if done:
            env.reset()

    """
     actions = [[0, 0, 0, 0] for _ in range(34)]
    while env_t != 25:
        actions = [[0, 0, 0.2, 0] for _ in range(34)]

        obs, reward, done, info = env.step(actions, env_t)
        # 循环画图
        env.recorder(env_t)
        render.render3D()
        plt.pause(0.001)
        env_t += 1

    actions = [[0, 0, 0, 0] for _ in range(34)]
    done = False
    while not done:
        for match_pair in match_pairs:
            i_uav = match_pair[0]
            i_aim = match_pair[1]
            uav = (env.state[i_uav][0], env.state[i_uav][1])
            aim = (get_points_x(i_aim), get_points_y(i_aim))
            vx, vy = go_to(uav, aim)
            actions[i_uav] = [vx, vy, 0, 0]
            if is_arrive(env.state[i_uav][:3], get_points(i_aim)):
                render.ax.scatter(env.state[i_uav][0], env.state[i_uav][1], env.state[i_uav][2], color='red', s=50)
        obs, reward, done, info = env.step(actions, env_t)
        # 循环画图
        env.recorder(env_t)
        render.render3D()
        plt.pause(0.001)
        env_t += 1
        if done:
            env.reset()
    """
    """
    
    """

    # done = False
    # while not done:
    #    print("ok")
    #    actions = [[0, 0, 0, 0] for _ in range(34)]
    #    env.recorder(env_t)
    #    render.render3D()
    #    plt.pause(0.001)
    #    env_t += 1
    """
                while env_t < 150:
            for i in range(32):
                actions[i] = [0, 0, -0.2, 0]
                obs, reward, done, info = env.step(actions, env_t)
                # 循环画图
                env.recorder(env_t)
                render.render3D()
                plt.pause(0.001)
                env_t += 1
        for i in range(32):
            render.ax.scatter(env.state[i][0], env.state[i][1], 0, color='red', s=50)
        """

    """
    正常绘图
    done = False
    while not done:
        actions = env.action_space.sample()  # sample an action
        obs, reward, done, info = env.step(actions, env_t)
        # 循环画图
        env.recorder(env_t)
        render.render3D()
        plt.pause(0.001)
        env_t += 1
        if done:
            env.reset()
    """


if __name__ == "__main__":
    main()
