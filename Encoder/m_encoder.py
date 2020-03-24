import numpy as np
from collections import Counter
from Encoder.fixedcode import dTob
from Encoder.intra_premode import *



class Encoder(object):
    """Defines the quantization layer."""
    def __init__(self):
        """
        # Arguments
            tensor: original feature tensor
            quantensor: quantized feature tensor
        """
        super(Encoder, self).__init__()
        #self.nBits = nBits

    def codeparameters(self, tensor, quantensor):
        """encode the parameters:channeil(C),min(V),max(V)

                # Arguments
                    tensor: original feature tensor: uesd to caculate min(V),max(V), obtain C
                    quantensor: uesd to obtian the most frequent feature values {m_i}
                """
        min_V = np.min(tensor)
        max_V = np.max(tensor)
        C = tensor.shape[0]                    #the numbers of channel
        list = self.array_frequency(quantensor)      #count the frequency of every feature value in the quabtensor
        #print(list)
        list_8 = list.most_common(8)           #get the list of the most frequent value and its amount
        m = [x[0] for x in list_8 ]            #get the 8 most frequency feature value
        # tile0 = quantensor[0:height, 0:width]
        # M = self.array_frequency(tile0)  # 统计每一个tile中特征值出现的概率，返回列表M,
        # for l1 in range(0,8):                                                                   #根据出现次数在每一个tile中对P进行排序
        #     for l2 in range(l1+1,8):
        #         #print(l1,l2)
        #         if M[m[l2]] > M[m[l1]]:
        #             tmp = m[l2]
        #             m[l2] = m[l1]
        #             m[l1] = tmp
        if len(m) < 8:
            for i in range(0, 8-len(m)):
                m.append(0)
        Cmin_V = dTob(min_V, pre=32)
        Cmax_V = dTob(max_V, pre=32)
        C2 = dTob(C, pre=0)
        Cm =[]
        #DCm =[]
        for i in range(0,len(m)):
            Cm.append(dTob(m[i], pre=32))


        ###plot the histogram of feature tensor
        #reduction=quantensor.reshape(-1)                         #把量化后的到的二维特征张量降维成一维数组，统计数组中各个数值出现的概率
        #reduction2=quantensor.flatten()
        #n, bins, patches = plt.hist(reduction, bins=128, normed=True, facecolor='#0504aa', alpha=0.7)
        #plt.savefig('exdata/unquant-histogram.png')
        #plt.show()
        #plt.axis('off')
        #plt.savefig('packetfeature/histogram.png')
        # 关闭当前显示的图像
        #plt.close()
        return [C2, Cmin_V, Cmax_V],Cm,m

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

    def vector_frequency(self, tensor):
        # reduction = tensor.reshape(-1)  # 把二维特征张量降维成一维数组，统计数组中各个数值出现的概率
        c = Counter()
        c = Counter(tensor)
        return c

    def encoder(self, tensor, m, width, height):
        """find a index from m for every feature map and coding index according a coding table

                                # Arguments
                                    tensor: the packed single featuremap
                                    p: a vector of the 8 most frequent feature value
                                    width: width of one tile
                                    heigth:height of one tile
                                # Returns

                                """
        H = tensor.shape[0]
        W = tensor.shape[1]
        k = 0
        tile={}
        residual = {}
        bp={}
        min_index = []
        Cindex = ''
        all_stream = ''
        for i in range(1,int(H/height)+1):
            for j in range(1,int(W/width)+1):
                tile[k] = tensor[(i - 1) * height:i * height, (j - 1) * width:j * width]                  #对组合好的特征图中的每一个tile进行统计
                error = []
                sum = 0
                for l in range(0, 8):                                                                    #从P中找出索引值
                    x = np.ones((height, width), np.int, 'C')
                    x = m[l] * x
                    x = abs(tile[k] - x)
                    error.append(x.sum())
                min_index.append(error.index(min(error)))
                residual[k], stream = self.codevalue(tile[k],m[error.index(min(error))])
                M = self.array_frequency(tile[k])  # 统计每一个tile中特征值出现的概率，返回列表M,
                for l1 in range(0, 8):  # 根据出现次数在每一个tile中对P进行排序
                    for l2 in range(l1 + 1, 8):
                        # print(l1,l2)
                        if M[m[l2]] > M[m[l1]]:
                            tmp = m[l2]
                            m[l2] = m[l1]
                            m[l1] = tmp
                k = k+1
                all_stream += stream
        for y in min_index:
            # Cindex.append(self.unarycode(y))
            Cindex += self.unarycode(y)
            Cindex += ' '
        #print(min_index)+++
        with open('./output/binary.txt', mode='a') as filename:
            filename.write(Cindex+all_stream)
        return Cindex, residual

    def unarycode(self, y):
        if y == 0:
            code = '1'
        elif y == 1:
            code = '10'
        elif y == 2:
            code = '110'
        elif y == 3:
            code = '1110'
        elif y == 4:
            code = '11110'
        elif y == 5:
            code = '111110'
        elif y == 6:
            code = '1111110'
        else:
            code = '1111111'
        return code

    def codevalue(self, tile, pj):
        block = []
        SATD = []
        (H, W) = tile.shape
        residual = np.zeros((H, W), np.int, 'C')  # F:列优先
        predict = np.zeros((H, W), np.int, 'C')
        neighbor_mode = []
        mode_index = np.zeros((int(H / 4), int(W / 4)))
        mode_stream = ''
        left_pad = np.array([0, 0, 0, 0])
        top_pad = np.array([0, 0, 0, 0])
        binary_value = ''
        binary_location = ''
        binary = ''
        for i in range(0, int(H / 4)):
            for j in range(0, int(W / 4)):
                i_0 = i * 4
                i_3 = (i + 1) * 4
                j_0 = j * 4
                j_3 = (j + 1) * 4
                block = tile[i_0:i_3, j_0:j_3]
                if j_0 == 0 and i_0 == 0:
                    top_pad[0] = top_pad[1] = top_pad[2] = top_pad[3] = pj
                    left_pad[0] = left_pad[1] = left_pad[2] = left_pad[3] = pj
                    top_left_pad = pj
                elif i_0 == 0 and j_0 != 0:
                    top_pad[0] = top_pad[1] = top_pad[2] = top_pad[3] = pj
                    left_pad[0] = tile[i_0][j_0 - 1]
                    left_pad[1] = tile[i_0 + 1][j_0 - 1]
                    left_pad[2] = tile[i_0 + 2][j_0 - 1]
                    left_pad[3] = tile[i_0 + 3][j_0 - 1]
                    top_left_pad = pj
                elif j_0 == 0 and i_0 != 0:
                    left_pad[0] = left_pad[1] = left_pad[2] = left_pad[3] = pj
                    top_pad[0] = tile[i_0 - 1][j_0]
                    top_pad[1] = tile[i_0 - 1][j_0 + 1]
                    top_pad[2] = tile[i_0 - 1][j_0 + 2]
                    top_pad[3] = tile[i_0 - 1][j_0 + 3]
                    top_left_pad = pj
                else:
                    top_pad[0] = tile[i_0 - 1][j_0]
                    top_pad[1] = tile[i_0 - 1][j_0 + 1]
                    top_pad[2] = tile[i_0 - 1][j_0 + 2]
                    top_pad[3] = tile[i_0 - 1][j_0 + 3]
                    left_pad[0] = tile[i_0][j_0 - 1]
                    left_pad[1] = tile[i_0 + 1][j_0 - 1]
                    left_pad[2] = tile[i_0 + 2][j_0 - 1]
                    left_pad[3] = tile[i_0 + 3][j_0 - 1]
                    top_left_pad = tile[i_0 - 1][j_0 - 1]
                index, predict[i_0:i_3, j_0:j_3], residual[i_0:i_3, j_0:j_3] = mode_select(block, top_pad, left_pad,
                                                                                           top_left_pad, pj)
                B = residual[i_0:i_3, j_0:j_3]
                # quant = QLayer(nbit)
                # v_quant = quant.bitQuantizer(V)
                mode_index[i][j] = index
                if i == 0 and j != 0:
                    neighbor_mode = [mode_index[i][j - 1]]
                    mpm = neighbor_mode[0]
                elif j == 0 and i != 0:
                    neighbor_mode = [mode_index[i - 1][j]]
                    mpm = neighbor_mode[0]
                elif j == 0 and i == 0:
                    mpm = 0
                else:
                    neighbor_mode = [mode_index[i][j - 1], mode_index[i - 1][j - 1],
                                     mode_index[i - 1][j]]  # 顺序要固定：左，左上，上
                    list = self.vector_frequency(neighbor_mode)
                    list_most = list.most_common(1)
                    mpm = int(list_most[0][0])
                mode_mpm, modecode = self.mode_index_code(mpm, index)
                mode_stream += mode_mpm
                mode_stream += modecode
                # model_dir = '/home/wangweiqian/yolov2.pytorch/CA_Entropy_Model/CA_EntropyModel_Test/model'
                # a = entropy(model_dir)
                # output_path = '/home/wangweiqian/yolov2.pytorch/my_mobile/output/compressed.bin'
                # a.entropy_encode(residual[i_0:i_3, j_0:j_3], output_path)
                stream1, stream2 = self.binarize(residual[i_0:i_3, j_0:j_3], index)  # 根据不同的index判断采用的预测模式，选择不同的扫描方式
                # stream1, stream2 = binarize(block, index)  # 根据不同的index判断采用的预测模式，选择不同的扫描方式
                binary_value += stream1
                binary_location += stream2
                binary += stream2
                binary += stream1
        #print(mode_index)
        # with open('/home/wangweiqian/yolov2.pytorch/my_mobile/output/binary.txt', mode='a') as filename:
        #     filename.write(mode_stream + binary)
        return residual, mode_stream + binary

    def mode_index_code(self, mpm, index):
        """
        对选择的模式下标进行编码,
        当前区域的预测模式等于mpm时,mode_mpm编码为1,不等时编码为0,且由00,01,10,11表示其余四种模式
        :param mpm:根据相邻左、左上、上侧4x4区域的预测模式统计出的最可能的模式
        :param index:当前区域选择的预测模式
        :return:表示模式类别的码字
        """
        if index == mpm:
            mode_mpm = '1'
            mode_code = ''
        else:
            mode_mpm = '0'
            if mpm == 0:
                if index == 1:
                    mode_code = '00'
                elif index == 2:
                    mode_code = '01'
                elif index == 3:
                    mode_code = '10'
                else:
                    mode_code = '11'
            elif mpm == 1:
                if index == 0:
                    mode_code = '00'
                elif index == 2:
                    mode_code = '01'
                elif index == 3:
                    mode_code = '10'
                else:
                    mode_code = '11'
            elif mpm == 2:
                if index == 0:
                    mode_code = '00'
                elif index == 1:
                    mode_code = '01'
                elif index == 3:
                    mode_code = '10'
                else:
                    mode_code = '11'
            elif mpm == 3:
                if index == 0:
                    mode_code = '00'
                elif index == 1:
                    mode_code = '01'
                elif index == 2:
                    mode_code = '10'
                else:
                    mode_code = '11'
            else:
                if index == 0:
                    mode_code = '00'
                elif index == 1:
                    mode_code = '01'
                elif index == 2:
                    mode_code = '10'
                else:
                    mode_code = '11'
        return mode_mpm, mode_code

    def binarize(self, block, mode):
        binary = ''
        if np.all(block == 0):
            skip = '1'
            location_indicator = skip
        else:
            skip = '0'
            if mode == 0 or mode == 3 or mode == 4:
                # print('zig-zag')
                residual_vector, location = zig_zaga_scan(block)
            elif mode == 1:
                # print('vetical')
                residual_vector, location = vectical_scan(block)
            else:
                # print('horizontal')
                residual_vector, location = horizontal_scan(block)
            location_indicator = skip + location
        if skip == '0':
            l = len(residual_vector)
            Max = max(residual_vector)
            for i in range(0, l):
                if residual_vector[i] > 0:
                    binary += '1'
                elif residual_vector[i] == 0:
                    continue
                else:
                    binary += '0'
                    residual_vector[i] = abs(residual_vector[i])
                if residual_vector[i] < 2:
                    flag = '1'
                    binary += flag
                    # for j in range(0, residual_vector[i]):
                    #     binary += '1'
                    # binary += '0'
                    binary += str(residual_vector[i])
                # elif residual_vector[i] > 9:
                #     flag = '2'
                #     binary += flag
                #     # for j in range(0, residual_vector[i]):
                #     #     binary += '1'
                #     # binary += '0'
                #     binary += str(residual_vector[i])
                else:
                    flag = '0'
                    binary += flag
                    binary += golomb(residual_vector[i], 0)

            # print(unary)
            return binary, location_indicator
        else:
            return '', location_indicator









