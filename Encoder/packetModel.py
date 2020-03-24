import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt
import math
from PIL import Image
import scipy.misc

class PacketModel(object):
    """Convert data to packets"""
    def __init__(self, data):
        """
            # Arguments
                data: 4-D tensor to be packetized
                rowsPerPacket: number of rows of the feature map to be considered as one packet
        """
        super(PacketModel, self).__init__()
        #self.dataShape     = data.shape
        #self.packetSeq     = self.dataToPacket(data)

    def dataToPacket(self, data):
        """ Converts 4D tensor to 5D tensor of packets

        # Arguments
            data: 4D tensor

        # Returns
            5D tensor
        """

        # force the number of filters to be square
        # n = int(np.ceil(np.sqrt(data.shape[0])))
        C = data.shape[0]
        n = 2 ** (math.floor(1 / 2 * (math.log(C, 2))))   #高
        m = 2 ** (math.ceil(1 / 2 * (math.log(C, 2))))    #宽
        #print(n,m)
        data = data.reshape((n, m) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], m * data.shape[3]) + data.shape[4:])
        return data

    def packetToData(self, tensor, height, width):
        """Converts the packets back to original 4D tensor

        # Returns
            4D tensor
        """
        H = tensor.shape[0]
        W = tensor.shape[1]
        n = int(H / height)
        m = int(W / width)
        k = 0
        unpacked = []
        #unpacked = np.zeros(())
        tile = {}
        bp = {}
        min_index = []
        Cindex = []
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                tile[k] = tensor[(i - 1) * height:i * height, (j - 1) * width:j * width]  # 对组合好的特征图中的每一个tile进行统计
                #print(tile[k].shape)
                unpacked = np.append(unpacked, tile[k])
                k = k+1
        unpacked = unpacked.reshape(n*m, height, width)
        unpacked = unpacked.astype(np.int)
        #scipy.misc.imsave('/home/wangweiqian/yolov2.pytorch/mobile/output/quanted_feature.tif', data)
        return unpacked
