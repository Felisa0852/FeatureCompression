import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import torch
from torch.autograd import Variable


class QLayer(object):
    """Defines the quantization layer."""
    def __init__(self, nBits):
        """
        # Arguments
            nBits: number of bits of quantization
        """
        super(QLayer, self).__init__()
        self.nBits = nBits
        self.lownbit = nBits - 1
        self.lowernbit = nBits - 2

    def bitQuantizer(self, data, weight):
        """Quantizes the input data to the set number of bits.

        # Arguments
            data: data to be quantized
        """
        start_time = time.time()
        self.quanData = np.zeros((data.shape[0], data.shape[1], data.shape[2]), np.int, 'C')
        self.max = np.max(data)
        self.min = np.min(data)


        C = data.shape[0]
        H = data.shape[1]
        W = data.shape[2]

        #weight = np.transpose(weight)
        sorted_weight = np.sort(weight)
        sorted_weight_index = np.argsort(weight)
        threshold_weight = sorted_weight[int(C*(7/8))]
        # threshold_weight = sorted_weight[0]
        #

        quantstream = ''
        count0 = 0
        count1 = 0
        for l in range(0, C):
            tile = data[l]
            important = weight[l]
            if important < threshold_weight:
                self.quanData[l] = np.round(((tile - self.min) / (self.max - self.min)) * ((2 ** self.lownbit) - 1))  # .astype(self.typeSize)
                # self.quanData[l] = replace
                quantstream += '1'
                count1 += 1
                # reduction = tile.reshape(-1)  # 把量化后的到的二维特征张量降维成一维数组，统计数组中各个数值出现的概率
                # # reduction2=R.flatten()
                # n, bins, patches = plt.hist(reduction, bins=128, normed=True, facecolor='#0504aa', alpha=0.7)
                # plt.savefig('/home/wangweiqian/projects/DeepFeatureCompression/test/histogram/importance/low/histogram_LI{}.png'.format(l))
                # # plt.show()
                # plt.axis('on')
                # plt.close()
            else:
                self.quanData[l] = np.round(((tile - self.min) / (self.max - self.min)) * (
                        (2 ** self.nBits) - 1))  # .astype(self.typeSize)
                quantstream += '0'
                count0 += 1
                # reduction = tile.reshape(-1)  # 把量化后的到的二维特征张量降维成一维数组，统计数组中各个数值出现的概率
                # # reduction2=R.flatten()
                # n, bins, patches = plt.hist(reduction, bins=128, normed=True, facecolor='#0504aa', alpha=0.7)
                # plt.savefig('/home/wangweiqian/projects/DeepFeatureCompression/test/histogram/importance/high/histogram_HI{}.png'.format(l))
                # # plt.show()
                # plt.axis('on')
                # plt.close()
        self.quanData = self.quanData.astype(np.int)
        return self.quanData, quantstream

    def  inverseQuantizer(self, data, maxF, minF, Qstream):
    # def inverseQuantizer(self, data, maxF, minF):
        """Performs inverse of quantization

        # Returns
            De-Quantized data.
        """
        C = data.shape[0]
        H = data.shape[1]
        W = data.shape[2]
        self.de_quanData = np.zeros((data.shape[0], data.shape[1], data.shape[2]), np.float, 'C')
        k = 0
        for l in range(0, C):
            tile = data[l]
            if Qstream[k] == '1':
                k += 1
                self.de_quanData[l] = (tile * (maxF - minF) / ((2 ** self.lownbit) - 1)) + minF
            else:
                k += 1
                # self.de_quanData[l][i_0:i_3, j_0:j_3] = (block * (maxF - minF) / ((2 ** self.lownbit) - 1)) + minF
                self.de_quanData[l] = (tile * (maxF - minF) / ((2 ** self.nBits) - 1)) + minF
        return self.de_quanData

    def array_frequency(self, tensor):
        """count the frequency of every feature value in the tensor

                        # Arguments
                            tensor: featuremap
                        # Returns
                            c2: a list of every value's frequency
                        """
        reduction = tensor.reshape(-1)  # 把二维特征张量降维成一维数组，统计数组中各个数值出现的概率
        c = Counter()
        c = Counter(reduction)
        return c

    def SF(self, A):
        M = A.shape[0]
        N = A.shape[1]
        minF = np.min(A)
        maxF = np.max(A)
        #A = A/(maxF-minF)
        RF = 0.0
        for i in range(0, M):
            for j in range(1, N):
                RF = RF + (A[i, j] - A[i, j-1])**2
        RF = (RF/(M*N))**(1/2)

        CF = 0.0
        for i in range(1,M):
            for j in range(0,N):
                CF = CF + (A[i, j] - A[i-1, j])**2
        CF = (CF/(M*N))**(1/2)
        SF = ((RF**2 + CF**2))**(1/2)
        return SF

    # def SF(self, A):
    #     # M = A.shape[0]
    #     # N = A.shape[1]
    #     # RF = 0.0
    #     # for i in range(0, M):
    #     #     for j in range(1, N):
    #     #         RF = RF + (A[i, j] - A[i, j - 1]) ** 2
    #     # RF = (RF / (M * N)) ** (1 / 2)
    #     #
    #     # CF = 0.0
    #     # for i in range(1, M):
    #     #     for j in range(0, N):
    #     #         CF = CF + (A[i, j] - A[i - 1, j]) ** 2
    #     # CF = (CF / (M * N)) ** (1 / 2)
    #     # SF = ((RF ** 2 + CF ** 2)) ** (1 / 2)
    #     H = A.shape[0]
    #     W = A.shape[1]
    #     AH1 = (torch.from_numpy(np.eye(H, k=1, dtype=int))).type(torch.FloatTensor)
    #     AH0 = (torch.from_numpy(np.eye(H, k=0, dtype=int))).type(torch.FloatTensor)
    #     D_H = AH0 - AH1
    #     AW1 = (torch.from_numpy(np.eye(W, k=1, dtype=int))).type(torch.FloatTensor)
    #     AW0 = (torch.from_numpy(np.eye(W, k=0, dtype=int))).type(torch.FloatTensor)
    #     D_W = Variable(AW0 - AW1).cuda()
    #     F_x = A * D_W
    #     F_y = Variable(np.transpose(D_H)).cuda() * A
    #     Z = 1 / 2 * (F_x + F_y)
    #     SF = np.linalg.norm(Z, ord=1)
    #     return SF



