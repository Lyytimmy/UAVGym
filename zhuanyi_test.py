import matplotlib.pyplot as plt
uav_init_state_zhuanyi = [[5, 18, 0, 0, 0, 0, 0],
                          [6, 16, 0, 0, 0, 0, 0],
                          [2, 24, 0, 0, 0, 0, 0],
                          [5, 22, 0, 0, 0, 0, 0],
                          [5, 14, 0, 0, 0, 0, 0],
                          [5, 26, 0, 0, 0, 0, 0],
                          [6, 10, 0, 0, 0, 0, 0],
                          [10, 30, 0, 0, 0, 0, 0],
                          [40, 32, 0, 0, 0, 0, 0],
                          [30, 34, 0, 0, 0, 0, 0],
                          [32, 34, 0, 0, 0, 0, 0],
                          [32, 36, 0, 0, 0, 0, 0],
                          [26, 42, 0, 0, 0, 0, 0],
                          [30, 42, 0, 0, 0, 0, 0],
                          [38, 46, 0, 0, 0, 0, 0],
                          [42, 44, 0, 0, 0, 0, 0],
                          [8, 25, 0, 0, 0, 0, 0],
                          [10, 15, 0, 0, 0, 0, 0],
                          [16, 15, 0, 0, 0, 0, 0],
                          [14, 25, 0, 0, 0, 0, 0],
                          [20, 6, 0, 0, 0, 0, 0],
                          [25, 3, 0, 0, 0, 0, 0],
                          [22, 10, 0, 0, 0, 0, 0],
                          [24, 15, 0, 0, 0, 0, 0],
                          [28, 30, 0, 0, 0, 0, 0],
                          [26, 32, 0, 0, 0, 0, 0],
                          [32, 30, 0, 0, 0, 0, 0],
                          [30, 35, 0, 0, 0, 0, 0],
                          [34, 20, 0, 0, 0, 0, 0],
                          [30, 25, 0, 0, 0, 0, 0],
                          [38, 30, 0, 0, 0, 0, 0],
                          [45, 25, 0, 0, 0, 0, 0]]

match_pairs_zhuanyi = [[0, [5, 18, 0], [40, 18, 5]], #
                       [1, [6, 16, 0], [36, 16, 5]], #
                       [2, [2, 24, 0], [35, 24, 5]], #
                       [3, [5, 22, 0], [35, 22, 5]],  #
                       [4, [5, 14, 0], [35, 16, 5]], #
                       [5, [5, 26, 0], [35, 26, 5]],  #
                       [6, [6, 10, 0], [35, 28, 5]],
                       [7, [10, 30,0], [35, 30, 5]],  #
                       [8, [40, 32, 0], [5, 32, 5]],  #
                       [9, [30, 34, 0], [5, 34, 5]],  #
                       [10, [32, 34, 0], [5, 36, 5]],
                       [11, [32, 36, 0], [5, 38, 5]],
                       [12, [26, 42, 0], [5, 40, 5]],
                       [13, [30, 42, 0], [5, 42, 5]],  #
                       [14, [38, 46, 0], [5, 44, 5]],
                       [15, [5, 40, 0], [5, 43, 5]],
                       [16, [8, 25, 0], [10, 35, 6]],
                       [17, [10, 15, 0], [12, 35, 6]],
                       [18, [16, 15, 0], [14, 35, 6]],
                       [19, [14, 25, 0], [16, 35, 6]],
                       [20, [20, 6, 0], [18, 35, 6]],
                       [21, [25, 3, 0], [20, 35, 6]],
                       [22, [22, 10, 0], [22, 35, 6]],  #
                       [23, [14, 5, 0], [16, 5, 6]],  #
                       [24, [10, 5, 0], [12, 5, 6]],
                       [25, [26, 32, 0], [28, 5, 6]],#
                       [26, [24, 30, 0], [26, 5, 6]],
                       [27, [30, 35, 0], [32, 5, 6]],#
                       [28, [34, 20, 0], [34, 5, 6]], #
                       [29, [30, 30, 0], [30, 5, 6]],
                       [30, [38, 30, 0], [38, 5, 6]],  #
                       [31, [45, 25, 0], [40, 5, 6]]] #

