import csv
import random
import sys
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
from array_data_zhuanyi import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from array_data_WH import *
from WH_test import *
from matplotlib.image import imread
import matplotlib.style as mplstyle
from zhuanyi_test import *

# 设置全局小数保留位数为2
np.set_printoptions(precision=2)
mplstyle.use('fast')


class UAVEnv(gym.Env):
    def __init__(self, uav_num,  map_w, map_h, map_z, Init_state):
        super(UAVEnv, self).__init__()
        self.uav_num = uav_num
        self.map_w = map_w
        self.map_h = map_h
        self.map_z = map_z
        self.position_pool = [[] for _ in range(self.uav_num)]
        self.state = Init_state

        # Define action and observation space  动作空间是所有动作状态的最大最小值之间、观察空间是所有状态的最大最小值之间
        self.action_space = spaces.Box(low=np.array([-0.35, -0.35, -0.35, 0] * self.uav_num),
                                       high=np.array([0.35, 0.35, 0.35, 1] * self.uav_num), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -1, -1, -1, 0] * self.uav_num),
                                            high=np.array([self.map_w, self.map_h, self.map_z, 1, 1, 1, 1] *
                                                          self.uav_num), dtype=np.float32)

    def recorder(self, env_t):
        if env_t % 2 == 0:
            for i in range(self.uav_num):
                x, y, z = self.state[i][:3]
                position = [x, y, z, env_t]
                self.position_pool[i].append(position)

    def step(self, actions, env_t):
        actions = np.array(actions).reshape(self.uav_num, 4)
        for i in range(self.uav_num):
            # update state x，y，z位置更新为原来的加上偏移量；vx，vy，vz更新，
            self.state[i][0] += actions[i][0]  # uav_x = vx*t, suppose t=1
            self.state[i][1] += actions[i][1]  # uav_y = vy*t
            self.state[i][2] += actions[i][2]  # uav_z = vz*t
            self.state[i][3:6] = actions[i][:3]  # update vx, vy, vz
            self.state[i][6] = actions[i][3]  # update sensor status
        return self.state, 0, False, {}

    def reset(self):
        self.state = np.zeros((self.uav_num, 7), dtype=np.float32)
        return self.state


class Render:
    def __init__(self, uav_num, state, buildings, map_w, map_h, map_z, uav_r, position_pool):
        self.uav_num = uav_num
        self.state = state
        self.buildings = buildings
        self.map_w = map_w
        self.map_h = map_h
        self.map_z = map_z
        self.uav_r = uav_r
        self.position_pool = position_pool
        self.line = []

        # 创建画布
        self.fig = plt.figure(figsize=(self.map_w, self.map_h))  # 设置画布大小
        self.ax = self.fig.add_subplot(111, projection='3d')  # 创建三维坐标系

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
        for i in range(self.uav_num):
            x_traj, y_traj, z_traj, _ = zip(*self.position_pool[i])
            l = self.ax.plot(x_traj[-10:], y_traj[-10:], z_traj[-10:], color='gray', alpha=0.7, linewidth=2.0)
            self.line.append(l)
        while len(self.line) > self.uav_num:
            old_line = self.line.pop(0)
            old_line[0].remove()

class SetConfig:
    def __init__(self, name):
        self.name = name
        self.uav_num = 0
        self.uav_r = 0.3
        self.map_w, self.map_h, self.map_z = 0, 0, 0
        self.buildings_location = []
        self.buildings = []
        self.match_pairs = []
        self.Init_state = []

    def Setting(self):
        if self.name == 'Map1':
            self.uav_num = 50
            self.map_w, self.map_h, self.map_z = 50, 50, 5
            self.buildings_location = buildings_location_WH
            self.buildings = buildings_WH
            self.match_pairs = match_pairs
            self.Init_state = uav_init_state
        elif self.name == 'Map2':
            self.uav_num = 32
            self.map_w, self.map_h, self.map_z = 50, 50, 5
            self.buildings_location = buildings_location_zhuanyi
            self.buildings = buildings_zhuanyi
            self.match_pairs = match_pairs_zhuanyi
            self.Init_state = uav_init_state_zhuanyi
        else:
            print("参数错误")
            sys.exit()

        return self.uav_num, self.map_w, self.map_h, self.map_z, self.buildings_location, self.buildings, self.match_pairs, self.uav_r, self.Init_state


class MvController:
    def __init__(self, map_w, map_h, map_z, buildings_location):
        self.map_w = map_w
        self.map_h = map_h
        self.map_z = map_z
        self.buildings_location = buildings_location

    def Move_up(self):
        return 0, 0, 0.2

    def Move_down(self):
        return 0, 0, -0.2

    def Move_to(self, uav, aim):
        max_speed = 0.3
        volatility = 0.02
        x_diff = uav[0] - aim[0]
        y_diff = uav[1] - aim[1]
        z_diff = uav[2] - aim[2]
        distance = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
        if abs(x_diff) < 0.1:
            vx = 0
        else:
            vx_normalized = x_diff / distance
            vx = vx_normalized * max_speed + random.gauss(0, volatility)
        if abs(y_diff) < 0.1:
            vy = 0
        else:
            vy_normalized = y_diff / distance
            vy = vy_normalized * max_speed + random.gauss(0, volatility)
        if abs(z_diff) < 0.1:
            vz = 0
        else:
            vz_normalized = z_diff / distance
            vz = vz_normalized * max_speed + random.gauss(0, volatility)
        return vx, vy, vz

    def Is_arrive(self, uav, aim):
        tolerance = 0.1
        x_error = abs(uav[0] - aim[0])
        y_error = abs(uav[1] - aim[1])
        z_error = abs(uav[2] - aim[2])
        return x_error < tolerance and y_error < tolerance and z_error < tolerance

    # def Is_collision(self):检测无人机之间是否会发生碰撞

    def Will_enter_buildings(self, uav, action, uav_r):
        next_x = uav[0] + action[0]
        next_y = uav[1] + action[1]
        next_z = uav[2] + action[2]
        grid_x = int(next_x)
        grid_y = int(next_y)
        height = self.buildings_location[grid_x][grid_y]
        if next_z - uav_r <= height:
            return True
        return False

    def Is_outside_map(self, uav, action):
        next_x = uav[0] + action[0]
        next_y = uav[1] + action[1]
        next_z = uav[2] + action[2]
        if next_x < 0 or next_x >= self.map_w or next_y < 0 or next_y >= self.map_h or next_z < 0 or next_z >= self.map_z:
            return True

        return False


def main():
    Map_name = 'Map2'
    env_t = 0
    # 初始化MAP模块
    MAP = SetConfig(Map_name)
    uav_num, map_w, map_h, map_z, buildings_location, buildings, match_pairs, uav_r, Init_state = MAP.Setting()
    # 初始化Env模块
    env = UAVEnv(uav_num,  map_w, map_h, map_z, Init_state)
    # 初始化render模块
    render = Render(uav_num, env.state, buildings, map_w, map_h, map_z, uav_r, env.position_pool)
    # 初始化MVController模块
    mvcontroller = MvController(map_w, map_h, map_z, buildings_location)
    # 开始
    actions = [[0, 0, 0, 0] for _ in range(uav_num)]
    flag = [False] * uav_num
    done = False
    while not done:
        for pair in match_pairs:
            index = pair[0]
            uav_state = env.state[index][:3]
            aim = pair[2]
            vx, vy, vz = mvcontroller.Move_to(uav_state, aim)
            if mvcontroller.Is_arrive(uav_state, aim):
                if not flag[index]:
                    flag[index] = True
                    render.ax.scatter(uav_state[0], uav_state[1], uav_state[2], color='red', s=50)
            if mvcontroller.Is_outside_map(uav_state, [vx, vy, vz]):
                vx, vy, vz = 0, 0, 0
            if mvcontroller.Will_enter_buildings(uav_state, [vx, vy, vz], uav_r):
                vx, vy, vz = mvcontroller.Move_up()
            actions[index] = [vx, vy, vz, 0]
        obs, reward, done, info = env.step(actions, env_t)
        env.recorder(env_t)
        render.render3D()
        plt.pause(0.01)
        env_t += 1
        if done:
            env.reset()


if __name__ == "__main__":
    main()
