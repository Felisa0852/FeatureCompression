#-*-coding:utf-8-*-
from torchvision.utils import save_image
from Encoder.entropy_coding import entropy_encode
from Encoder.entropy_coding import entropy_decode
from Encoder.intra_premode import Dec2Bin
from Decoder.decoder import *
from collections import Counter
# from cloud.mirror import HalfMirror
import argparse
import time
from tools.tests import prepare_im_data
# from cloud.detectNet import Detect
from tools.yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import numpy as np
from PIL import Image
from torch.autograd import Variable
from Encoder.packetModel import PacketModel
import math
import torch
import torch.nn as nn
from TaskNet.mirrorNet import HalfMirror, Mirror
import tools.pytorch_ssim
from Encoder.fixedcode import bTod
import cv2 as cv
import matplotlib.pyplot as plt
from Encoder.my_quantizer import QLayer
from Encoder.m_encoder import Encoder
from TaskNet.se_module import SELayer
from TaskNet.detectNet_w import Detect
from torchvision import transforms
import os


feat_result = []
weight_result = []
feat_result1 = []
weight_result1 = []


def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default='yolov2_epoch_160', type=str)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


def quant_packet(V, nbit, weight):
    """Quantize the featur tensor V, And the combined into a single feature map.

             # Arguments
                V: A feature tensor with (C,H,W)
                nbit: turn the 32bit float feature value number to nbit(n= 6,8,12)
            # Returns
                 packetfeature: A single feature map after tiled
    """
    #调用量化层
    quant = QLayer(nbit)
    v_quant, quantstream = quant.bitQuantizer(V, weight)


    pack =PacketModel(v_quant)
    packetfeature =pack.dataToPacket(v_quant)
    return v_quant, packetfeature, quantstream


def get_features_hook(self, input, output):
    feat_result.append(output.data.cpu().numpy())


def get_weight_hook(self, input, output):
    weight_result.append(output[0].data.cpu().numpy())
    feat_result.append(output[1].data.cpu().numpy())
    # weight_result.append(output)


def get_weight_hook1(self, input, output):
    weight_result1.append(output[0].data.cpu().numpy())
    feat_result1.append(output[1].data.cpu().numpy())
    # weight_result.append(output)


def vector_frequency(tensor):
    c = Counter()
    c = Counter(tensor)
    return c


def array_frequency(tensor):
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


def unarydecode(code, C):
    l = len(code)
    i = 0
    d_index = []
    count = 0
    k = 0
    while k < C:
        count += 1
        i += 1
        if code[i] == ' ':
            if count == 7 and code[i-1] == '1':
                d_index.append(count)
            else:
                d_index.append(count - 1)
            k += 1
            count = 0
            i += 1
    # if code == '1':
    #     y = 0
    # elif code == '10':
    #     y = 1
    # elif code == '110':
    #     y = 2
    # elif code == '1110':
    #     y = 3
    # elif code == '11110':
    #     y = 4
    # elif code == '111110':
    #     y = 5
    # elif code == '1111110':
    #     y = 6
    # else:
    #     y = 7
    # return y
    return d_index, i


def psnr(img1, img2):
    MSE = nn.MSELoss(size_average=True, reduce=True, reduction='elementwise_mean')
    mse = MSE(img1, img2)
    if mse == 0:
        return 100
    #PIXEL_MAX = 255.0
    PIXEL_MAX = 1
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    #PSNR = 10 * math.log10((PIXEL_MAX**2) / mse)
    return  PSNR


def compression(V_featensor, quant_bit, height, width, weight):
    start_time = time.time()
    quant_feature, packed_feature, Qstream = quant_packet(V_featensor, quant_bit, weight)

    # cv.imwrite('/home/wangweiqian/yolov2.pytorch/my_mobile/output/quanted_feature_100.jpg', packed_feature,
    #            [int(cv.IMWRITE_JPEG_QUALITY), 100])
    # cv.imwrite('/home/wangweiqian/yolov2.pytorch/my_mobile/output/quanted_feature_98.jpg', packed_feature,
    #            [int(cv.IMWRITE_JPEG_QUALITY), 98])
    # cv.imwrite('/home/wangweiqian/yolov2.pytorch/my_mobile/output/quanted_feature_95.jpg', packed_feature,
    #            [int(cv.IMWRITE_JPEG_QUALITY), 95])
    # cv.imwrite('/home/wangweiqian/yolov2.pytorch/my_mobile/output/quanted_feature_92.jpg', packed_feature,
    #            [int(cv.IMWRITE_JPEG_QUALITY), 92])
    # cv.imwrite('/home/wangweiqian/yolov2.pytorch/my_mobile/output/quanted_feature_80.jpg', packed_feature,
    #            [int(cv.IMWRITE_JPEG_QUALITY), 80])
    # cv.imwrite('/home/wangweiqian/yolov2.pytorch/mobile/output/quanted_feature.png', packed_feature)

    maxF = np.max(V_featensor)
    minF = np.min(V_featensor)
    #print(maxF, minF)
    # pack0 = PacketModel(packed_feature)
    # unpacked_feature0 = pack0.packetToData(packed_feature, height, width)
    # de_quant0 = QLayer(quant_bit)
    # dequan_feature0 = de_quant0.inverseQuantizer(unpacked_feature0, maxF, minF)

    # coding the quantized_feature map
    c = Encoder()

    # 1.coding parameters ：channel numbers C,min,max
    parameters1, parameters2, pp = c.codeparameters(V_featensor, packed_feature)  # parameters1:[C,min,max]  parameters2:coded P


    open('./output/binary.txt', 'w').close()
    with open('./output/binary.txt', mode='a') as filename:
        filename.write(Qstream)
    indexcode= c.encoder(packed_feature, pp, height, width)
    input_path = "./output/binary.txt"
    output_path = './output/compressed.bin'
    entropy_encode(input_path, output_path)
    total_time = time.time() - start_time
    print("compression complete in {}s!!".format(total_time))
    return parameters1, parameters2, pp, indexcode, Qstream, quant_feature


def decompression(inputpath, outputpath, parameters1, parameters2, height, width, quant_bit):
    start_time = time.time()
    C = bTod(parameters1[0], 32)
    minF = bTod(parameters1[1], 32)
    maxF = bTod(parameters1[2], 32)
    #print(minF,maxF)
    entropy_decode(inputpath, outputpath)
    input = open(outputpath, 'r')
    decode_stream = input.read()
    p0 = [0,0,0,0,0,0,0,0]
    for i in range(0,8):
        p0[i] = bTod(parameters2[i],32)
    dQstream = decode_stream[0:C]
    index_stream = decode_stream[C:len(decode_stream)]
    d_index, index_len = unarydecode(index_stream, C)
    binary_stream = decode_stream[C + index_len:len(decode_stream)]

    l = len(binary_stream)
    location_index = 0
    index = 0
    decode_residual_vector = []
    decode_residual_all = {}
    left_pad = np.array([0, 0, 0, 0])
    top_pad = np.array([0, 0, 0, 0])

    k = 0
    tile_list = {}
    n = 2 ** (math.floor(1 / 2 * (math.log(C, 2))))  # 高
    m = 2 ** (math.ceil(1 / 2 * (math.log(C, 2))))  # 宽
    H = n * height
    W = m * width
    featuremap_d = np.zeros((H, W), np.int, 'C')

    while index < l:
        tile_num = 0
        pd_index = []
        while tile_num < C:
            block_num = 0
            mode_count = 0
            #d_index = unarydecode(indexcode[tile_num])
            #pd_index.append(d_index)
            # print(indexcode[block_num], d_index)
            pj = p0[d_index[tile_num]]
            while block_num < ((height / 4) * (width / 4)):
            # while block_num < ((n / 4) * (m / 4)):
                if binary_stream[index] == '1':
                    index += 1
                    mode_count += 1
                else:
                    index += 3
                    mode_count += 3
                block_num += 1
            mode_stream_d = binary_stream[index - mode_count:index]

            ##########特征值解码#####
            block_num = 0
            while block_num < ((height / 4) * (width / 4)):
            # while block_num < ((n / 4) * (m / 4)):
                if binary_stream[index] == '1':
                    index += 1
                    decode_residual_vector = np.zeros(16, np.int, 'C')
                    decode_residual_all[block_num] = decode_residual_vector
                    block_num += 1
                    decode_residual_vector = []
                else:
                    index += 1
                    location_stream = binary_stream[index:index + 16]
                    index += 16
                    location_l = len(location_stream)
                    mode_l = len(mode_stream_d)
                    loopnum = 0
                    while loopnum < 16:
                        if location_stream[location_index] == '0':
                            decode_residual_vector.append(0)
                            location_index += 1
                        else:
                            if binary_stream[index] == '1':
                                sign = 1
                                index += 1
                            else:
                                sign = 0
                                index += 1
                            if binary_stream[index] == '1':
                                flag = 1
                                index += 1
                                # count = 0
                                # for j in range(index, l):
                                #     if binary_stream[j] == '0':
                                #         break
                                #     count += 1
                                # index = index + count + 1
                                s = binary_stream[index]
                                index += 1
                                if sign == 1:
                                    # decode_residual_vector.append(count)
                                    decode_residual_vector.append(int(s))
                                else:
                                    # decode_residual_vector.append(-count)
                                    decode_residual_vector.append(-(int(s)))
                                location_index += 1
                            # elif binary_stream[index] == '2':
                            #     flag = 2
                            #     index += 1
                            #     # count = 0
                            #     # for j in range(index, l):
                            #     #     if binary_stream[j] == '0':
                            #     #         break
                            #     #     count += 1
                            #     # index = index + count + 1
                            #     s = binary_stream[index:index + 1]
                            #     index += 2
                            #     if sign == 1:
                            #         # decode_residual_vector.append(count)
                            #         decode_residual_vector.append(int(s))
                            #     else:
                            #         # decode_residual_vector.append(-count)
                            #         decode_residual_vector.append(-(int(s)))
                            #     location_index += 1
                            else:
                                flag = 0
                                index += 1
                                count = 0
                                for j in range(index, l):
                                    if binary_stream[j] == '1':
                                        break
                                    count += 1
                                index += count
                                s = binary_stream[index:index + count + 1]
                                v = int(s, 2)
                                v = v - 1
                                s = Dec2Bin(v)
                                index = index + count + 1
                                if sign == 1:
                                    decode_residual_vector.append(int(s, 2))
                                else:
                                    # decode_residual_vector.append(-(int(s)))
                                    bb = -(int(s, 2))
                                    decode_residual_vector.append(-(int(s, 2)))
                                location_index += 1
                        loopnum += 1
                    decode_residual_all[block_num] = decode_residual_vector
                    loopnum = 0
                    decode_residual_vector = []
                    location_index = 0
                    block_num += 1
            # print(decode_residual_all)

            ###########预测模式解码########
            k = 0
            str_index = 0  # 预测模式码流的下标
            mode_index_d = np.zeros((int(height / 4), int(width / 4)), np.int, 'C')
            tile_residual_d = np.zeros((height, width), np.int, 'C')
            feature_value = np.zeros((height, width), np.int, 'C')
            # mode_index_d = np.zeros((int(n / 4), int(m / 4)), np.int, 'C')
            # tile_residual_d = np.zeros((n, m), np.int, 'C')
            # feature_value = np.zeros((n, m), np.int, 'C')
            for i in range(0, int(height / 4)):
                for j in range(0, int(width / 4)):
            # for i in range(0, int(n / 4)):
            #     for j in range(0, int(m / 4)):
                    i_0 = i * 4
                    i_3 = (i + 1) * 4
                    j_0 = j * 4
                    j_3 = (j + 1) * 4
                    # residual_block = residual[i_0:i_3, j_0:j_3]
                    if i == 0 and j != 0:
                        neighbor_mode = [mode_index_d[i][j - 1]]
                        mpm = neighbor_mode[0]
                    elif j == 0 and i != 0:
                        neighbor_mode = [mode_index_d[i - 1][j]]
                        mpm = neighbor_mode[0]
                    elif j == 0 and i == 0:
                        mpm = 0
                    else:
                        neighbor_mode = [mode_index_d[i][j - 1], mode_index_d[i - 1][j - 1],
                                         mode_index_d[i - 1][j]]  # 顺序要固定：左，左上，上
                        list = vector_frequency(neighbor_mode)
                        list_most = list.most_common(1)
                        mpm = int(list_most[0][0])
                    mode_mpm = mode_stream_d[str_index]

                    if mode_mpm == '0':
                        modecode = mode_stream_d[str_index + 1:str_index + 3]
                        str_index = str_index + 3
                    else:
                        modecode = ''
                        str_index = str_index + 1

                    mode_index_d[i][j] = mode_index_decode(mode_mpm, mpm, modecode)
                    block_d = vector2block(decode_residual_all[k], mode_index_d[i][j])

                    if j_0 == 0 and i_0 == 0:
                        top_pad[0] = top_pad[1] = top_pad[2] = top_pad[3] = pj
                        left_pad[0] = left_pad[1] = left_pad[2] = left_pad[3] = pj
                        top_left_pad = pj
                    elif i_0 == 0 and j_0 != 0:
                        top_pad[0] = top_pad[1] = top_pad[2] = top_pad[3] = pj
                        left_pad[0] = feature_value[i_0][j_0 - 1]
                        left_pad[1] = feature_value[i_0 + 1][j_0 - 1]
                        left_pad[2] = feature_value[i_0 + 2][j_0 - 1]
                        left_pad[3] = feature_value[i_0 + 3][j_0 - 1]
                        top_left_pad = pj
                    elif j_0 == 0 and i_0 != 0:
                        left_pad[0] = left_pad[1] = left_pad[2] = left_pad[3] = pj
                        top_pad[0] = feature_value[i_0 - 1][j_0]
                        top_pad[1] = feature_value[i_0 - 1][j_0 + 1]
                        top_pad[2] = feature_value[i_0 - 1][j_0 + 2]
                        top_pad[3] = feature_value[i_0 - 1][j_0 + 3]
                        top_left_pad = pj
                    else:
                        top_pad[0] = feature_value[i_0 - 1][j_0]
                        top_pad[1] = feature_value[i_0 - 1][j_0 + 1]
                        top_pad[2] = feature_value[i_0 - 1][j_0 + 2]
                        top_pad[3] = feature_value[i_0 - 1][j_0 + 3]
                        left_pad[0] = feature_value[i_0][j_0 - 1]
                        left_pad[1] = feature_value[i_0 + 1][j_0 - 1]
                        left_pad[2] = feature_value[i_0 + 2][j_0 - 1]
                        left_pad[3] = feature_value[i_0 + 3][j_0 - 1]
                        top_left_pad = feature_value[i_0 - 1][j_0 - 1]
                    decode_value = residual_decode(block_d, mode_index_d[i][j], top_pad, left_pad, top_left_pad, pj)
                    tile_residual_d[i_0:i_3, j_0:j_3] = block_d
                    feature_value[i_0:i_3, j_0:j_3] = decode_value
                    k += 1
            MM = array_frequency(feature_value)  # 统计每一个tile中特征值出现的概率，返回列表M,
            # for l in range(0, 7):
            #     if MM[p0[l + 1]] > MM[p0[l]]:
            #         tmp = p0[l]
            #         p0[l] = p0[l + 1]
            #         p0[l + 1] = tmp
            #
            for l1 in range(0, 8):  # 根据出现次数在每一个tile中对P进行排序
                for l2 in range(l1+1, 8):
                    #print(l1, l2)
                    if MM[p0[l2]] > MM[p0[l1]]:
                        tmp = p0[l2]
                        p0[l2] = p0[l1]
                        p0[l1] = tmp
            tile_list[tile_num] = feature_value
            tile_num += 1
        #print(pd_index)
        k = 0
        for i in range(int(H / height)):
            for j in range(int(W / width)):
        # for i in range(int(H / n)):
        #     for j in range(int(W / m)):
                featuremap_d[i * height:(i + 1) * height, j * width:(j + 1) * width] = tile_list[k]
                # featuremap_d[i * n:(i + 1) * n, j * m:(j + 1) * m] = tile_list[k]
                k += 1
    pack = PacketModel(featuremap_d)
    unpacked_feature = pack.packetToData(featuremap_d, height, width)
    de_quant = QLayer(quant_bit)
    dequan_feature = de_quant.inverseQuantizer(unpacked_feature, maxF, minF, dQstream)
    end_time = time.time()
    decompress_time = end_time - start_time
    print('decompression cost time:', decompress_time, 's')
    return dequan_feature, featuremap_d, unpacked_feature


def to_img(x):
    # h = im_info['height']
    # w = im_info['width']
    x = x.view(3, 416, 416)
    # x = x.view(3, 416, 416)
    # im_data = x.resize((h, w))
    return x


def importance_based_compression():
    # ====================load model=====================
    # model1:encoder to get intermediate deep feature
    model1 = HalfMirror()
    pretrained = torch.load('./models/Reconstruction.pkl')
    model1.load_state_dict(pretrained['model'])
    model1.cuda()
    model1.eval()

    # model2:decoder to reconstruct input image
    model2 = Mirror()
    pretrained_dict = model1.state_dict()
    model_dict = model2.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    model2.load_state_dict(model_dict)
    model2.cuda()
    model2.eval()

    # model3:decoder to detect objection
    model3 = Detect()
    pretrained = torch.load('./models/Detection.pth')
    model3.load_state_dict(pretrained['model'])
    model3.cuda()
    model3.eval()

    # ====================input image and extract feature====================
    # input the picture
    img = Image.open('./data/23.tif')
    im_data, im_info = prepare_im_data(img)
    im_data_variable = Variable(im_data).cuda()

    # get the feature tensor from split point max_7
    handle_feat = model1.max_7.register_forward_hook(get_features_hook)  # conv1
    a = model1(im_data_variable)
    handle_feat.remove()
    feat1 = feat_result[0]
    V_featensor = feat1[0, ...]

    # get the importance fator
    handle_weight = model1.channel_select.register_forward_hook(get_weight_hook)
    a = model1(im_data_variable)
    handle_weight.remove()
    weight1 = weight_result[0]
    weight = np.transpose(weight1[0])[0][0]

    # # ====================intermediate deep feature compression====================
    C = V_featensor.shape[0]
    height = V_featensor.shape[1]
    width = V_featensor.shape[2]
    quant_bit = 4
    parameters1, parameters2, pp, indexcode, Qstream, quant_feature = compression(V_featensor, quant_bit, height, width,
                                                                                      weight)

    #
    # ====================intermediate deep feature decompression====================
    inputpath = './output/compressed.bin'
    outputpath = './output/decompressed.txt'
    decode_feature, featuremap_d, unpacked_feature = decompression(inputpath, outputpath, parameters1, parameters2,
                                                                       height, width, quant_bit)
    compressedsize = os.path.getsize(inputpath)
    print('compressed_size:{}'.format(compressedsize))
    # ====================complete multi_tasks====================
    inputfeature = decode_feature.reshape(1, C, height, width)
    inputfeature = torch.from_numpy(inputfeature)
    inputfeature = inputfeature.type(torch.FloatTensor)  # 不变
    data_variable = Variable(inputfeature).cuda()

    handle_feat = model2.channel_select.register_forward_hook(get_weight_hook1)
    output = model2(data_variable)
    handle_feat.remove()
    detection_input = feat_result1[0]
    detection_input = torch.from_numpy(detection_input)
    detection_input = detection_input.type(torch.FloatTensor)  # 不变
    data_variable2 = Variable(detection_input).cuda()

    #       ==============input reconstruction===========

    ssim_loss = tools.pytorch_ssim.ssim(im_data_variable, output)
    psnr_loss = psnr(im_data_variable, output)
    pic = to_img(output.cpu().data)
    pic_in = to_img(im_data_variable.cpu().data)
    save_image(pic_in, './output/imagein.png')
    save_image(pic, './output/imageout.png')
    print('SSIM {}  PSNR {}'.format(ssim_loss.data.item(), psnr_loss))

    #      ==============object detection================
    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    tic = time.time()
    yolo_output = model3(data_variable2)

    yolo_output = [item[0].data for item in yolo_output]

    detections = yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4)
    print(detections)
    if len(detections) == 0:
        plt.figure()
        plt.imshow(img)
        plt.show()
    else:
        toc = time.time()
        cost_time = toc - tic
        print('im detect, cost time {:4f}, FPS: {}'.format(
            toc - tic, int(1 / cost_time)))

        det_boxes = detections[:, :5].cpu().numpy()
        det_classes = detections[:, -1].long().cpu().numpy()

        im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=classes)
        plt.figure()
        plt.imshow(im2show)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('./output/detect_result.jpg', dpi = 150)
        plt.show()


importance_based_compression()