import random
import matplotlib.pyplot as plt
import numpy as np

x1 = 10
x2 = 12
x3 = 14
x4 = 16
x5 = 18
x6 = 20
x7 = 22
x8 = 24
x9 = 26
x10 = 28
x11 = 30
x12 = 32
x13 = 36
x14 = 36
x15 = 38
x16 = 40
x17 = 42
y1 = 29
y2 = 27
y3 = 23
y4 = 21
y5 = 20
y6 = 18
y7 = 15
y8 = 12
y9 = 10
z = 5
# Given 32 points with their x and y coordinates
point = {
    'point_1': (x4, y1, z),
    'point_2': (x9, y1, z),
    'point_3': (x13, y1, z),
    'point_4': (x2, y2, z),
    'point_5': (x6, y2, z),
    'point_6': (x12, y2, z),
    'point_7': (x16, y2, z),
    'point_8': (x1, y3, z),
    'point_9': (x5, y3, z),
    'point_10': (x7, y3, z),
    'point_11': (x11, y3, z),
    'point_12': (34, y3, z),
    'point_13': (x17, y3, z),
    'point_14': (x6, y4, z),
    'point_15': (x12, y4, z),
    'point_16': (x4, y5, z),
    'point_17': (x8, y5, z),
    'point_18': (x10, y5, z),
    'point_19': (x14, y5, z),
    'point_20': (x2, y6, z),
    'point_21': (x6, y6, z),
    'point_22': (x12, y6, z),
    'point_23': (x16, y6, z),
    'point_24': (x3, y7, z),
    'point_25': (x9, y7, z),
    'point_26': (x15, y7, z),
    'point_27': (x4, y8, z),
    'point_28': (x8, y8, z),
    'point_29': (x10, y8, z),
    'point_30': (x14, y8, z),
    'point_31': (x6, y9, z),
    'point_32': (x12, y9, z),
    'point_33': (x4, 16, z),
    'point_34': (x14, 16, z)
}

points = [
    (x4, y1, z), (x9, y1, z), (x13, y1, z),
    (x2, y2, z), (x6, y2, z), (x12, y2, z), (x16, y2, z),
    (x1, y3, z), (x5, y3, z), (x7, y3, z), (x11, y3, z), (34, y3, z), (x17, y3, z),
    (x6, y4, z), (x12, y4, z),
    (x4, y5, z), (x8, y5, z), (x10, y5, z), (x14, y5, z),
    (x2, y6, z), (x6, y6, z), (x12, y6, z), (x16, y6, z),
    (x3, y7, z), (x9, y7, z), (x15, y7, z),
    (x4, y8, z), (x8, y8, z), (x10, y8, z), (x14, y8, z),
    (x6, y9, z), (x12, y9, z),
    (x4, 16, z), (x14, 16, z)
]

uav_init_pos = [[9, 6, 0], [7, 3, 0], [9, 11, 0], [1, 11, 0], [10, 11, 0], [10, 8, 0], [10, 4, 0], [4, 9, 0],
                [48, 8, 0], [47, 3, 0], [39, 7, 0], [38, 4, 0], [38, 5, 0], [43, 12, 0], [44, 5, 0], [42, 7, 0],
                [39, 2, 0], [2, 40, 0], [12, 40, 0], [1, 37, 0], [9, 48, 0], [3, 47, 0], [7, 40, 0], [12, 41, 0],
                [8, 47, 0], [45, 48, 0], [38, 39, 0], [49, 39, 0], [37, 48, 0], [37, 46, 0], [43, 35, 0], [47, 47, 0],
                [42, 49, 0], [40, 40, 0]]
uav_init_state = [[9, 6, 0, 0, 0, 0, 0], [7, 3, 0, 0, 0, 0, 0], [9, 11, 0, 0, 0, 0, 0], [1, 11, 0, 0, 0, 0, 0],
                  [10, 11, 0, 0, 0, 0, 0], [10, 8, 0, 0, 0, 0, 0], [10, 4, 0, 0, 0, 0, 0], [4, 9, 0, 0, 0, 0, 0],
                  [48, 8, 0, 0, 0, 0, 0], [47, 3, 0, 0, 0, 0, 0], [39, 7, 0, 0, 0, 0, 0], [38, 4, 0, 0, 0, 0, 0],
                  [38, 5, 0, 0, 0, 0, 0], [43, 12, 0, 0, 0, 0, 0], [44, 5, 0, 0, 0, 0, 0], [42, 7, 0, 0, 0, 0, 0],
                  [39, 2, 0, 0, 0, 0, 0], [2, 40, 0, 0, 0, 0, 0], [12, 40, 0, 0, 0, 0, 0], [1, 37, 0, 0, 0, 0, 0],
                  [9, 48, 0, 0, 0, 0, 0], [3, 47, 0, 0, 0, 0, 0], [7, 40, 0, 0, 0, 0, 0], [12, 41, 0, 0, 0, 0, 0],
                  [8, 47, 0, 0, 0, 0, 0], [45, 48, 0, 0, 0, 0, 0], [38, 39, 0, 0, 0, 0, 0], [49, 39, 0, 0, 0, 0, 0],
                  [37, 48, 0, 0, 0, 0, 0], [37, 46, 0, 0, 0, 0, 0], [43, 35, 0, 0, 0, 0, 0], [47, 47, 0, 0, 0, 0, 0],
                  [42, 49, 0, 0, 0, 0, 0], [40, 40, 0, 0, 0, 0, 0]]
uav_5_pos = [[9, 6, 5], [7, 3, 5], [9, 11, 5], [1, 11, 5], [10, 11, 5], [10, 8, 5], [10, 4, 5], [4, 9, 5], [48, 8, 5],
             [47, 3, 5], [39, 7, 5], [38, 4, 5], [38, 5, 5], [43, 12, 5], [44, 5, 5], [42, 7, 5], [39, 2, 5],
             [2, 40, 5], [12, 40, 5], [1, 37, 5], [9, 48, 5], [3, 47, 5], [7, 40, 5], [12, 41, 5], [8, 47, 5],
             [45, 48, 5], [38, 39, 5], [49, 39, 5], [37, 48, 5], [37, 46, 5], [43, 35, 5], [47, 47, 5], [42, 49, 5],
             [40, 40, 5]]
match_pairs = [[0, 26], [1, 23], [2, 19], [3, 7], [4, 32], [5, 30], [6, 27], [7, 15], [8, 25], [9, 29], [10, 31],
               [11, 33], [12, 28], [13, 22], [14, 18], [15, 21], [16, 24], [17, 3], [18, 0], [19, 4], [20, 1], [21, 8],
               [22, 9], [23, 13], [24, 5], [25, 2], [26, 6], [27, 12], [28, 11], [29, 10], [30, 14], [31, 17], [32, 16],
               [33, 20]]


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
            x, y = generate_uav_pos(1, 14, 1, 14)
            if not is_collision(uav_pos[:i], x, y, uav_r):
                break
        uav_pos[i].append(x)
        uav_pos[i].append(y)
        uav_pos[i].append(z)

    # Group 2: 9 UAVs with x in the range [35, 49] and y in the range [0, 15]
    for i in range(8, 17):
        while True:
            x, y = generate_uav_pos(36, 48, 1, 14)
            if not is_collision(uav_pos[:i], x, y, uav_r):
                break
        uav_pos[i].append(x)
        uav_pos[i].append(y)
        uav_pos[i].append(z)

    # Group 3: 8 UAVs with x in the range [0, 15] and y in the range [35, 49]
    for i in range(17, 25):
        while True:
            x, y = generate_uav_pos(1, 14, 36, 48)
            if not is_collision(uav_pos[:i], x, y, uav_r):
                break
        uav_pos[i].append(x)
        uav_pos[i].append(y)
        uav_pos[i].append(z)
    # Group 4: 9 UAVs with x in the range [35, 49] and y in the range [35, 49]
    for i in range(25, 34):
        while True:
            x, y = generate_uav_pos(35, 49, 35, 49)
            if not is_collision(uav_pos[:i], x, y, uav_r):
                break
        uav_pos[i].append(x)
        uav_pos[i].append(y)
        uav_pos[i].append(z)

    return uav_pos


def find_matching_pairs(uav_positions, target_positions):
    num_uavs = len(uav_positions)
    num_targets = len(target_positions)

    # Create an array to store the matching pairs
    matching_pairs = []

    # Create arrays to keep track of whether a UAV or target is already matched
    uav_matched = [False] * num_uavs
    target_matched = [False] * num_targets

    # Sort target positions based on distances from each UAV
    sorted_target_indices = np.argsort(
        [np.linalg.norm(np.array(uav_positions) - np.array(target_positions), axis=1) for uav_pos in uav_positions])

    # Loop through each UAV
    for uav_idx in range(num_uavs):
        # Find the nearest target that is not already matched
        min_distance = float('inf')
        min_target_idx = None

        for target_idx in sorted_target_indices[uav_idx]:
            if not target_matched[target_idx]:
                distance = np.linalg.norm(np.array(uav_positions[uav_idx]) - np.array(target_positions[target_idx]))
                if distance < min_distance:
                    min_distance = distance
                    min_target_idx = target_idx

        # If a valid target is found, add the matching pair
        if min_target_idx is not None:
            matching_pairs.append([uav_idx, min_target_idx])
            uav_matched[uav_idx] = True
            target_matched[min_target_idx] = True

    return matching_pairs


# array = generate_uav_pos_array()
# print(array)
# matching_pairs = find_matching_pairs(uav_5_pos, points)
# print(matching_pairs)

""" 

# Extract x and y coordinates from the points dictionary
x_values = [point[0] for point in points.values()]
y_values = [point[1] for point in points.values()]

# Plot the points
plt.scatter(x_values, y_values, color='blue', label='Points')

# Add labels to the points
for label, (x, y, z) in points.items():
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

# Set plot labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Plot of 32 Points')

# Show the legend and the plot
plt.legend()
plt.grid()
plt.show()
"""
