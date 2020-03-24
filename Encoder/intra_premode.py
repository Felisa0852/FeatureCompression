import numpy as np


def mode_select(block, top_pad, left_pad, top_left_pad, pj):
    """
    计算不同预测模式下的残差，选择出使码长最短的模式。并返回残差值
    :param block:4x4区域的特征值
    :param top_pad:相邻上侧的特征值
    :param left_pad:相邻左侧的特征值
    :param top_left_pad:左上的特征值
    :param pj:
    :return:选择后的4x4区域的残差值
    """
    residual = {}
    predict = {}
    SATD = [1, 1, 1, 1, 1]
    predict[0], residual[0] = mode0(block, pj)
    predict[1], residual[1] = mode1(block, top_pad)
    predict[2], residual[2] = mode2(block, left_pad)
    predict[3], residual[3] = mode3(block, top_pad, left_pad, top_left_pad)
    predict[4], residual[4] = mode4(block, top_pad, left_pad, top_left_pad)

    for i in range(0, 5):
        SATD[i] = cal_SATD(residual[i])

    mode_index = int(SATD.index(min(SATD)))
    index = int(mode_index)
    select_residual = residual[index]
    select_predict = predict[index]
    return index, select_predict, select_residual


def mode0(block, pj):
    """
    预测模式0：所有值由索引值pj预测
    :param block: 4x4区域的特征值
    :param pj:该图块的索引值
    :return:4x4区域的残差值
    """
    x = np.ones((4, 4), np.int, 'C')
    predict = pj * x
    residual = block - predict
    # print(predict)
    return predict, residual


def mode1(block, top_pad):
    """
    预测模式1：垂直预测模式
    :param block: 4x4区域的特征值
    :param top_pad:相邻上侧的特征值
    :return:4x4区域的残差值
    """
    predict = np.zeros((4, 4), np.int, 'C')
    for l in range(0, 4):
        predict[0, l] = top_pad[l]

    for m in range(1, 4):
        for n in range(0, 4):
            predict[m, n] = block[m - 1, n]
    residual = block - predict
    # print(predict)
    return predict, residual


def mode2(block, left_pad):
    """
    预测模式2：水平预测模式
    :param block: 4x4区域的特征值
    :param left_pad: 相邻左侧的特征值
    :return: 4x4区域的残差值
    """

    predict = np.zeros((4, 4), np.int, 'C')
    for l in range(0, 4):
        predict[l, 0] = left_pad[l]

    for m in range(1, 4):
        for n in range(0, 4):
            predict[n, m] = block[n, m - 1]
    residual = block - predict
    # print(predict)
    return predict, residual


def mode3(block, top_pad, left_pad, top_left_pad):
    """
    预测模式3：滤波方式,系数1/32[3,7,22](top-left,top,left)
    :param block: 4x4区域的特征值
    :param top_pad: 相邻上侧的特征值
    :param left_pad: 相邻左侧的特征值
    :param top_left_pad:左上的特征值
    :return:4x4区域的残差值
    """

    predict = np.zeros((4, 4), np.int, 'C')
    predict[0, 0] = 1 / 32 * (3 * top_left_pad + 7 * top_pad[0] + 22 * left_pad[0])
    # predict[0, 0] = np.round(1 / 32 * (3 * top_left_pad + 7 * top_pad[0] + 22 * left_pad[0]))
    for l in range(1, 4):
        top_left = top_pad[l - 1]
        top = top_pad[l]
        left = block[0][l - 1]
        predict[0, l] = 1 / 32 * (3 * top_left + 7 * top + 22 * left)
        # predict[0, l] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left))

    for k in range(1, 4):
        top_left = left_pad[k - 1]
        top = block[k - 1][0]
        left = left_pad[k]
        predict[k, 0] = 1 / 32 * (3 * top_left + 7 * top + 22 * left)
        # predict[k, 0] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left))

    for m in range(1, 4):
        for n in range(1, 4):
            top_left = block[m - 1][n - 1]
            top = block[m - 1][n]
            left = block[m][n - 1]
            predict[m, n] = 1 / 32 * (3 * top_left + 7 * top + 22 * left)
            # predict[m, n] = np.round(1 / 32 * (3 * top_left + 7 * top + 22 * left))

    residual = block - predict
    # print(predict)
    return predict, residual


def mode4(block, top_pad, left_pad, top_left_pad):
    """
    预测模式4：滤波方式,系数1/32[14,0,18](top-left,top,left)
    :param block: 4x4区域的特征值
    :param top_pad: 相邻上侧的特征值
    :param left_pad: 相邻左侧的特征值
    :param top_left_pad:左上的特征值
    :return:4x4区域的残差值
    """
    predict = np.zeros((4, 4), np.int, 'C')
    predict[0, 0] = 1 / 32 * (14 * top_left_pad + 0 * top_pad[0] + 18 * left_pad[0])
    # predict[0, 0] = np.round(1 / 32 * (14 * top_left_pad + 0 * top_pad[0] + 18 * left_pad[0]))
    for l in range(1, 4):
        top_left = top_pad[l - 1]
        top = top_pad[l]
        left = block[0][l - 1]
        predict[0, l] = 1 / 32 * (14 * top_left + 0 * top + 18 * left)
        # predict[0, l] = np.round(1 / 32 * (14 * top_left + 0 * top + 18 * left))

    for k in range(1, 4):
        top_left = left_pad[k - 1]
        top = block[k - 1][0]
        left = left_pad[k]
        predict[k, 0] = 1 / 32 * (14 * top_left + 0 * top + 18 * left)
        # predict[k, 0] = np.round(1 / 32 * (14 * top_left + 0 * top + 18 * left))

    for m in range(1, 4):
        for n in range(1, 4):
            top_left = block[m - 1][n - 1]
            top = block[m - 1][n]
            left = block[m][n - 1]
            predict[m, n] = 1 / 32 * (14 * top_left + 0 * top + 18 * left)
            # predict[m, n] = np.round(1 / 32 * (14 * top_left + 0 * top + 18 * left))
    residual = block - predict
    # print(predict)
    return predict, residual


def cal_SATD(residual):
    H = np.array([[1, 1, 1, 1],
                  [1, -1, 1, -1],
                  [1, 1, -1, -1],
                  [1, -1, -1, 1]])
    R = H * residual * H
    # print(R)
    SATD_value = (abs(R)).sum()
    # print(SATD_value)
    return SATD_value


def zig_zaga_scan(matrix):
    """Scan matrix of zigzag algorithm"""
    vector = []
    location_indicator = ''
    n = len(matrix) - 1

    i = 0
    j = 0

    for _ in range(n * 2):
        vector.append(matrix[i][j])
        if matrix[i][j] == 0:
            location_indicator += '0'
        else:
            location_indicator += '1'

        if j == n:   # right border
            i += 1     # shift
            while i != n:   # diagonal passage
                vector.append(matrix[i][j])
                if matrix[i][j] == 0:
                    location_indicator += '0'
                else:
                    location_indicator += '1'

                i += 1
                j -= 1
        elif i == 0:  # top border
            j += 1
            while j != 0:
                vector.append(matrix[i][j])
                if matrix[i][j] == 0:
                    location_indicator += '0'
                else:
                    location_indicator += '1'

                i += 1
                j -= 1
        elif i == n:   # bottom border
            j += 1
            while j != n:
                vector.append(matrix[i][j])
                if matrix[i][j] == 0:
                    location_indicator += '0'
                else:
                    location_indicator += '1'

                i -= 1
                j += 1
        elif j == 0:   # left border
            i += 1
            while i != 0:
                vector.append(matrix[i][j])
                if matrix[i][j] == 0:
                    location_indicator += '0'
                else:
                    location_indicator += '1'

                i -= 1
                j += 1

    vector.append(matrix[i][j])
    if matrix[i][j] == 0:
        location_indicator += '0'
    else:
        location_indicator += '1'

    return vector,location_indicator


def vectical_scan(matrix):
    vector = []
    location_indicator = ''
    for i in range(0,4):
        for j in range(0,4):
            vector.append(matrix[i][j])
            if matrix[i][j] == 0:
                location_indicator += '0'
            else:
                location_indicator += '1'
    return vector,location_indicator


def horizontal_scan(matrix):
    vector = []
    location_indicator = ''
    for i in range(0,4):
        for j in range(0,4):
            vector.append(matrix[j][i])
            if matrix[j][i] == 0:
                location_indicator += '0'
            else:
                location_indicator += '1'
    return vector,location_indicator


def golomb(value, k):
    binary = Dec2Bin(value)
    #print(binary)
    l = len(binary)
    binary = binary[0:l-k]
    #print(binary)
    binary =addBinary(binary,'1')
    #print(binary)
    prefix_num = len(binary)
    for i in range(0, prefix_num-1):
        binary = '0' + binary
    #print(binary)
    for i in range(0,k):
        binary += '0'
    #print(binary)
    return binary


def Dec2Bin(dec):
    result = ''
    if dec:
        result = Dec2Bin(dec // 2)
        return result + str(dec % 2)
    else:
        return result


def addBinary(a, b):
    """
    :type a: str
    :type b: str
    :rtype: str
    """
    if len(a) < len(b):  # 填充"0"，将两个字符串变为等长
        temp = a
        a = b
        b = temp
    a = a[::-1]  # 倒序二进制字符串
    b = b[::-1]
    while len(a) != len(b):  # 二进制字符串长度设置相同
        b = b + "0"
    extra = 0  # 进位
    new_binary = ""
    for index, num in enumerate(a):  # 遍历
        b_sum = int(b[index])
        new_binary = new_binary + str((int(num) + b_sum + extra) % 2)  # 二进制加法运算
        if int(num) + b_sum + extra > 1:  # 是否进位
            extra = 1
        else:
            extra = 0
    if extra == 1:  # 最高位是否进位
        new_binary = new_binary + "1"
    return new_binary[::-1]  # 倒序输出


