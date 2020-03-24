#-*-coding:utf-8-*-
import math
import numpy as np


def vector2block(vector,mode_index):
    if mode_index == 0 or mode_index == 3 or mode_index == 4:
        residual_block = d_zigzag(vector)
    elif mode_index == 1:
        residual_block = d_vectical(vector)
    else:
        residual_block = d_horizontal(vector)
    return residual_block


def d_zigzag(vector):
    l = len(vector)
    size = int(math.sqrt(l))
    matrix = np.zeros((size, size), np.int, 'C')
    n = size - 1
    i = 0
    j = 0
    k = 0
    #for _ in range(size):
    while k < (l-1):
        matrix[i][j] = vector[k]
        k+=1
        if j == n:  # right border
            i += 1  # shift
            while i != n:  # diagonal passage
                matrix[i][j] = vector[k]
                k += 1
                i += 1
                j -= 1
        elif i == 0:  # top border
            j += 1
            while j != 0:
                matrix[i][j] = vector[k]
                k += 1

                i += 1
                j -= 1
        elif i == n:  # bottom border
            j += 1
            while j != n:
                matrix[i][j] = vector[k]
                k += 1
                i -= 1
                j += 1
        elif j == 0:  # left border
            i += 1
            while i != 0:
                matrix[i][j] = vector[k]
                k += 1
                i -= 1
                j += 1
    matrix[i][j] = vector[k]
    k += 1

    return matrix


def d_vectical(vector):
    l = len(vector)
    size = int(math.sqrt(l))
    matrix = np.zeros((size, size), np.int, 'C')
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            matrix[i][j] = vector[k]
            k += 1
    return matrix


def d_horizontal(vector):
    l = len(vector)
    size = int(math.sqrt(l))
    matrix = np.zeros((size, size), np.int, 'C')
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            matrix[j][i] = vector[k]
            k += 1
    return matrix


def mode_index_decode(mode_mpm,mpm,modecode):
    """
    解码的出当前区域选择的预测模式
    :param mode_mpm: 是否为最可能模式
    :param mpm: 最可能模式
    :param modecode: 表示模式类别的码字
    :return:当前区域的预测模式
    """
    if mode_mpm == '1':
        mode_index_d = mpm
    else:
        if mpm == 0:
            if modecode == '00':
                mode_index_d = 1
            elif modecode == '01':
                mode_index_d = 2
            elif modecode == '10':
                mode_index_d = 3
            else:
                mode_index_d = 4
        elif mpm == 1:
            if modecode == '00':
                mode_index_d = 0
            elif modecode == '01':
                mode_index_d = 2
            elif modecode == '10':
                mode_index_d = 3
            else:
                mode_index_d = 4
        elif mpm == 2:
            if modecode == '00':
                mode_index_d = 0
            elif modecode == '01':
                mode_index_d = 1
            elif modecode == '10':
                mode_index_d = 3
            else:
                mode_index_d = 4
        elif mpm == 3:
            if modecode == '00':
                mode_index_d = 0
            elif modecode == '01':
                mode_index_d = 1
            elif modecode == '10':
                mode_index_d = 2
            else:
                mode_index_d = 4
        else:
            if modecode == '00':
                mode_index_d = 0
            elif modecode == '01':
                mode_index_d = 1
            elif modecode == '10':
                mode_index_d = 2
            else:
                mode_index_d = 3
    return mode_index_d


def residual_decode(residual,index_d,top_pad, left_pad, top_left_pad,pj):
    """
    根据预测模式解码出4x4区域特征的真实值
    :param residual:残差值
    :param index_d:预测模式
    :param top_pad:相邻上侧的特征值
    :param left_pad:相邻左侧的特征值
    :param top_left_pad:左上的特征值
    :param pj:
    :return:真实值
    """
    if index_d == 0:
        real_value = dmode0(residual, pj)
    elif index_d == 1:
        real_value = dmode1(residual, top_pad)
    elif index_d == 2:
        real_value = dmode2(residual, left_pad)
    elif index_d == 3:
        real_value = dmode3(residual, top_pad, left_pad, top_left_pad)
    else:
        real_value = dmode4(residual, top_pad, left_pad, top_left_pad)
    return real_value


def dmode0(residual, pj):
    x = np.ones((4, 4), np.int, 'C')
    predict = pj * x
    real_value = residual + predict
    #print(predict)
    return real_value


def dmode1(residual, top_pad):
    predict = np.zeros((4, 4), np.int, 'C')
    real_value = np.zeros((4, 4), np.int, 'C')
    for l in range(0, 4):
        #predict[0, l] = top_pad[l]
        real_value[0, l] = top_pad[l] + residual[0, l]

    for m in range(1, 4):
        for n in range(0, 4):
            real_value[m, n] = residual[m, n]+real_value[m - 1, n]
    #print(predict)
    return real_value


def dmode2(residual, left_pad):
    real_value = np.zeros((4, 4), np.int, 'C')
    for l in range(0, 4):
        real_value[l, 0] = left_pad[l] + residual[l, 0]

    for m in range(1, 4):
        for n in range(0, 4):
            real_value[n, m] = real_value[n, m - 1] + residual[n, m]
    return real_value


def dmode3(residual, top_pad, left_pad, top_left_pad):
    real_value = np.zeros((4, 4), np.int, 'C')
    # real_value[0, 0] = np.round(1 / 32 * (3 * top_left_pad + 7 * top_pad[0] + 22 * left_pad[0]) + residual[0, 0])
    real_value[0, 0] = 1 / 32 * (3 * top_left_pad + 7 * top_pad[0] + 22 * left_pad[0]) + residual[0, 0]
    for l in range(1, 4):
        top_left = top_pad[l - 1]
        top = top_pad[l]
        left = real_value[0][l - 1]
        # real_value[0, l] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[0, l])
        real_value[0, l] = 1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[0, l]

    for k in range(1, 4):
        top_left = left_pad[k - 1]
        top = real_value[k - 1][0]
        left = left_pad[k]
        # real_value[k, 0] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[k, 0])
        real_value[k, 0] = 1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[k, 0]


    for m in range(1, 4):
        for n in range(1, 4):
            top_left = real_value[m - 1][n - 1]
            top = real_value[m - 1][n]
            left = real_value[m][n - 1]
            # real_value[m, n] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[m, n])
            real_value[m, n] = 1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[m, n]


    #print(predict)
    return real_value


def dmode4(residual, top_pad, left_pad, top_left_pad):
    real_value = np.zeros((4, 4), np.int, 'C')
    # real_value[0, 0] = np.round(1 / 32 * (14 * top_left_pad + 0 * top_pad[0] + 18 * left_pad[0])+ residual[0, 0])
    real_value[0, 0] = 1 / 32 * (14 * top_left_pad + 0 * top_pad[0] + 18 * left_pad[0]) + residual[0, 0]
    for l in range(1, 4):
        top_left = top_pad[l - 1]
        top = top_pad[l]
        left = real_value[0][l - 1]
        # real_value[0, l] = np.round(1 / 32 * (14 * top_left + 0 * top + 18 * left) + residual[0, l])
        real_value[0, l] = 1 / 32 * (14 * top_left + 0 * top + 18 * left) + residual[0, l]

    for k in range(1, 4):
        top_left = left_pad[k - 1]
        top = real_value[k - 1][0]
        left = left_pad[k]
        # real_value[k, 0] = np.round(1 / 32 * (14 * top_left + 0 * top + 18 * left) + residual[k, 0])
        real_value[k, 0] = 1 / 32 * (14 * top_left + 0 * top + 18 * left) + residual[k, 0]

    for m in range(1, 4):
        for n in range(1, 4):
            top_left = real_value[m - 1][n - 1]
            top = real_value[m - 1][n]
            left = real_value[m][n - 1]
            # real_value[m, n] = np.round(1 / 32 * (14 * top_left + 0 * top + 18 * left) + residual[m, n])
            real_value[m, n] = 1 / 32 * (14 * top_left + 0 * top + 18 * left) + residual[m, n]
    return real_value

# def dmode3(residual, top_pad, left_pad, top_left_pad):
#     real_value = np.zeros((4, 4), np.int, 'C')
#     # real_value[0, 0] = np.round(1 / 32 * (3 * top_left_pad + 7 * top_pad[0] + 22 * left_pad[0]) + residual[0, 0])
#     # real_value[0, 0] = 1 / 32 * (3 * top_left_pad + 7 * top_pad[0] + 22 * left_pad[0]) + residual[0, 0]
#     real_value[0, 0] = 1 / 4 * (2 * top_left_pad + top_pad[0] + left_pad[0]) + residual[0, 0]
#     for l in range(1, 4):
#         top_left = top_pad[l - 1]
#         top = top_pad[l]
#         left = real_value[0][l - 1]
#         # real_value[0, l] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[0, l])
#         # real_value[0, l] = 1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[0, l]
#         real_value[0, l] = 1 / 4 * (2 * top_left + top + left) + residual[0, l]
#
#     for k in range(1, 4):
#         top_left = left_pad[k - 1]
#         top = real_value[k - 1][0]
#         left = left_pad[k]
#         # real_value[k, 0] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[k, 0])
#         # real_value[k, 0] = 1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[k, 0]
#         real_value[k, 0] = 1 / 4 * (2 * top_left + top + left) + residual[k, 0]
#
#     for m in range(1, 4):
#         for n in range(1, 4):
#             top_left = real_value[m - 1][n - 1]
#             top = real_value[m - 1][n]
#             left = real_value[m][n - 1]
#             # real_value[m, n] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[m, n])
#             #real_value[m, n] = 1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[m, n]
#             real_value[m, n] = 1 / 4 * (2 * top_left + top + left) + residual[m, n]
#
#     #print(predict)
#     return real_value
#
#
# def dmode4(residual, top_pad, left_pad, top_left_pad):
#     real_value = np.zeros((4, 4), np.int, 'C')
#     # real_value[0, 0] = np.round(1 / 32 * (3 * top_left_pad + 7 * top_pad[0] + 22 * left_pad[0]) + residual[0, 0])
#     # real_value[0, 0] = 1 / 32 * (3 * top_left_pad + 7 * top_pad[0] + 22 * left_pad[0]) + residual[0, 0]
#     real_value[0, 0] = 1 / 4 * (top_left_pad + 2 * top_pad[0] + left_pad[0]) + residual[0, 0]
#     for l in range(1, 4):
#         top_left = top_pad[l - 1]
#         top = top_pad[l]
#         left = real_value[0][l - 1]
#         # real_value[0, l] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[0, l])
#         # real_value[0, l] = 1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[0, l]
#         real_value[0, l] = 1 / 4 * (top_left + 2 * top + left) + residual[0, l]
#
#     for k in range(1, 4):
#         top_left = left_pad[k - 1]
#         top = real_value[k - 1][0]
#         left = left_pad[k]
#         # real_value[k, 0] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[k, 0])
#         # real_value[k, 0] = 1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[k, 0]
#         real_value[k, 0] = 1 / 4 * (top_left + 2 * top + left) + residual[k, 0]
#
#     for m in range(1, 4):
#         for n in range(1, 4):
#             top_left = real_value[m - 1][n - 1]
#             top = real_value[m - 1][n]
#             left = real_value[m][n - 1]
#             # real_value[m, n] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[m, n])
#             #real_value[m, n] = 1 / 32 * (3 * top_left + 7 * top + 22 * left) + residual[m, n]
#             real_value[m, n] = 1 / 4 * (top_left + 2 * top + left) + residual[m, n]
#
#     #print(predict)
#     return real_value