import matplotlib.pyplot as plt
import random
import numpy as np

def Get_Uav_Pos(uav_id):
    global x, y, z
    pos = np.array([[x[uav_id]], [y[uav_id]], [z[uav_id]]])
    return pos


x = [1, 1, 1, 0.5, 0.5, 0.5, 0, 0, 0]
y = [1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0]
z = [0, 0, 0, 0, 0, 0, 0, 0, 0]
aim = [[10], [10], [10]]
point = []
# 参数设置
k_mig = 1
k_coh = 1
r_max = 5
k_sep = 7
v_cmd = np.zeros([3, 9])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(aim[0], aim[1], aim[2], s=50)
plt.ion()
for t in range(500):
    for i in range(9):
        pos_i = Get_Uav_Pos(i)
        r_mig = aim - pos_i
        v_mig = k_mig * r_mig / np.linalg.norm(r_mig)
        v_sep = np.zeros([3, 1])
        v_coh = np.zeros([3, 1])
        N_i = 0
        for j in range(9):
            if j != i:
                N_i += 1
                pos_j = Get_Uav_Pos(j)
                if np.linalg.norm(pos_i - pos_j) < r_max:
                    r_ij = pos_j - pos_i
                    v_sep += - k_sep * r_ij / np.linalg.norm(r_ij)
                    v_coh += k_coh * r_ij
        v_sep = v_sep / N_i
        v_coh = v_coh / N_i
        v_cmd[:, i:i+1] = v_sep + v_coh + v_mig
    for i in range(9):
        x[i] += v_cmd[0, i]
        y[i] += v_cmd[1, i]
        z[i] += v_cmd[2, i]
        point.append(ax.scatter(x[i], y[i], z[i], s=20))
        if len(point) > 9:
            old_point = point.pop(0)
            old_point.remove()
    plt.pause(1)