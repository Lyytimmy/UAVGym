import math
import matplotlib.pyplot as plt
#
# points = []
# r = 1
# center = [1, 1]
# for i in range(18):
#     theta = math.radians(90 - 20 * i)  # 转换为弧度
#     x = center[0] + r * math.cos(theta)
#     y = center[1] + r * math.sin(theta)
#     points.append([round(x, 2), round(y, 2)])  # 四舍五入保留2位小数
#
# points2 = []
# r = 0.5
# center = [1, 1]
# for i in range(6):
#     theta = math.radians(90 - 60 * i)  # 转换为弧度
#     x = center[0] + r * math.cos(theta)
#     y = center[1] + r * math.sin(theta)
#     points2.append([round(x, 2), round(y, 2)])  # 四舍五入保留2位小数
#
# print(points+points2)
import matplotlib.pyplot as plt

coordinates = []

def plot_points():
    plt.clf()
    for coord in coordinates:
        plt.plot(coord[0], coord[1], 'ro')
    plt.axis([0, 240, 0, 240])
    plt.draw()

def mouse_click(event):
    if event.xdata is not None and event.ydata is not None:
        coordinates.append([event.xdata, event.ydata])
        plot_points()

def key_press(event):
    if event.key == 'd':
        if coordinates:
            coordinates.pop()
            plot_points()
    elif event.key == 'r':
        plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 240])
ax.set_ylim([0, 240])
ax.set_title('Click to add points. Press "d" to delete the last point. Press "r" to end.')

cid = fig.canvas.mpl_connect('button_press_event', mouse_click)
cid2 = fig.canvas.mpl_connect('key_press_event', key_press)

plt.show()

# 格式化输出坐标点
formatted_coordinates = [[f"{x:.2f}", f"{y:.2f}"] for x, y in coordinates]
output = "[" + ",".join([f"[{x},{y}]" for x, y in formatted_coordinates]) + "]"
print(output)

