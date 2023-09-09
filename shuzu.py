import random


def generate_array():
    array = [[random.randint(0, 5) for _ in range(50)] for _ in range(50)]

    for i in range(50):
        if i in [8, 9, 10, 11, 12, 18, 19, 20, 22, 28, 29, 31, 38, 39, 40, 41, 42, 48, 49]:
            array[i] = [0] * 50
        for j in range(50):
            if j in [8, 9, 10, 11, 12, 18, 19, 21,  28, 29, 31, 32, 38, 39, 40, 41, 42, 48, 49]:
                array[i][j] = 0

    for row in array:
        for i in range(len(row)):
            if row[i] >= 4:
                row[i] = 0

    for row in array:
        for i in range(len(row)):
            if row[i] == 3:
                row[i] = 2
    return array


def print_array2(array):
    for row in array:
        print(' '.join(str(element) for element in row))


def print_array(array):
    print('[', end='')
    for i in range(len(array)):
        print('[', end='')
        print(','.join(str(element) for element in array[i]), end='')
        print(']', end='')
        if i != len(array) - 1:
            print(',', end='')
    print(']')


def modify_array(array, x1, x2, y1, y2, value):
    for i in range(x1, x2 + 1):
        for j in range(y1, y2 + 1):
            array[i][j] = value

    return array


# 生成数组
array = generate_array()
array = modify_array(array,0,15,0,49,0)
array = modify_array(array,35,49,0,49,0)
array = modify_array(array,0,49,0,15,0)
array = modify_array(array,0,49,35,49,0)
def print_array3(array):
    array_building = []
    for i in range(50):
        for j in range(50):
            building_height = array[i][j]
            array_building.append([[i, j, 0], [i, j + 1, 0], [i + 1, j, 0], [i + 1, j + 1, 0], building_height])
    print(array_building)


# 打印数组
# print_array3(array)
print_array3(array)
