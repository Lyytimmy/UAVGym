import matplotlib.pyplot as plt

uav_init_state_zhuanyi = [[47, 48, 0, 0, 0, 0, 0],
                          [35, 39, 0, 0, 0, 0, 0],
                          [44, 20, 0, 0, 0, 0, 0],
                          [42, 17, 0, 0, 0, 0, 0],
                          [39, 44, 0, 0, 0, 0, 0],
                          [35, 13, 0, 0, 0, 0, 0],
                          [42, 8, 0, 0, 0, 0, 0],
                          [44, 16, 0, 0, 0, 0, 0],
                          [19, 3, 0, 0, 0, 0, 0],
                          [13, 10, 0, 0, 0, 0, 0],
                          [5, 9, 0, 0, 0, 0, 0],
                          [40, 12, 0, 0, 0, 0, 0],
                          [41, 9, 0, 0, 0, 0, 0],
                          [14, 13, 0, 0, 0, 0, 0],
                          [27, 5, 0, 0, 0, 0, 0],
                          [39, 13, 0, 0, 0, 0, 0],
                          [17, 11, 0, 0, 0, 0, 0],
                          [25, 23, 0, 0, 0, 0, 0],
                          [4, 1, 0, 0, 0, 0, 0],
                          [25, 5, 0, 0, 0, 0, 0],
                          [16, 4, 0, 0, 0, 0, 0],
                          [23, 21, 0, 0, 0, 0, 0],
                          [5, 7, 0, 0, 0, 0, 0],
                          [10, 4, 0, 0, 0, 0, 0],
                          [7, 45, 0, 0, 0, 0, 0],
                          [15, 42, 0, 0, 0, 0, 0],
                          [48, 41, 0, 0, 0, 0, 0],
                          [28, 38, 0, 0, 0, 0, 0],
                          [43, 39, 0, 0, 0, 0, 0],
                          [12, 40, 0, 0, 0, 0, 0],
                          [49, 46, 0, 0, 0, 0, 0],
                          [10, 48, 0, 0, 0, 0, 0]]

match_pairs_zhuanyi = [[0, [47, 48, 0], [17, 17, 2.3]],
                       [1, [35, 39, 0], [17, 19, 2.3]],
                       [2, [44, 20, 0], [17, 21, 2.3]],
                       [3, [42, 17, 0], [17, 23, 2.3]],
                       [4, [39, 44, 0], [17, 25, 2.3]],
                       [5, [35, 13, 0], [17, 27, 2.3]],
                       [6, [42, 8, 0], [17, 29, 2.3]],
                       [7, [44, 16, 0], [17, 31, 2.3]],
                       [8, [19, 3, 0], [17, 33, 2.3]],
                       [9, [13, 10, 0], [19, 33, 2.3]],
                       [10, [5, 9, 0], [21, 33, 2.3]],
                       [11, [40, 12, 0], [23, 33, 2.3]],
                       [12, [41, 9, 0], [25, 33, 2.3]],
                       [13, [14, 13, 0], [27, 33, 2.3]],
                       [14, [27, 5, 0], [29, 33, 2.3]],
                       [15, [39, 13, 0], [31, 33, 2.3]],
                       [16, [17, 11, 0], [33, 33, 2.3]],
                       [17, [25, 23, 0], [33, 31, 2.3]],
                       [18, [4, 1, 0], [33, 29, 2.3]],
                       [19, [25, 5, 0], [33, 27, 2.3]],
                       [20, [16, 4, 0], [33, 25, 2.3]],
                       [21, [23, 21, 0], [33, 23, 2.3]],
                       [22, [5, 7, 0], [33, 21, 2.3]],
                       [23, [10, 4, 0], [33, 19, 2.3]],
                       [24, [7, 45, 0], [33, 17, 2.3]],
                       [25, [15, 42, 0], [31, 17, 2.3]],
                       [26, [48, 41, 0], [29, 17, 2.3]],
                       [27, [28, 38, 0], [27, 17, 2.3]],
                       [28, [43, 39, 0], [25, 17, 2.3]],
                       [29, [12, 40, 0], [23, 17, 2.3]],
                       [30, [49, 46, 0], [21, 17, 2.3]],
                       [31, [10, 48, 0], [19, 17, 2.3]]]

