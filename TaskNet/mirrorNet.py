from yolov2 import Yolov2
from de_darknet import de_Darknet19
from darknet import Darknet19


import numpy as np

import torch
import torch.nn as nn
from TaskNet.se_module import SELayer

from torch.autograd import Variable

"""SELayer"""


class HalfMirror(nn.Module):
    num_channel = 128

    def __init__(self, channel=None, weights_file=False):
        super(HalfMirror, self).__init__()
        if channel:
            self.num_channel = channel
        darknet = Darknet19()
        seblock = SELayer(self.num_channel)
        self.conv0 = darknet.layer0
        self.max_1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.conv2 = darknet.layer1
        self.max_3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.conv4_6 = darknet.layer2
        self.max_7 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        ###############################################################
        darknet19 = de_Darknet19()
        # self.deconv = nn.Sequential(darknet19.layer1,darknet19.layer2,darknet19.layer3,)
        # self.reconstruct = nn.Sequential(darknet19.layer4)
        #self.unpool_7 = nn.MaxUnpool2d(2, stride=2)
        self.channel_select = seblock
        self.deconv4_6 = darknet19.layer2
        #self.unpool_3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2 = darknet19.layer3
        #self.unpool_1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv0 = darknet19.layer4


    # def forward(self, x,index_1,index_3,index_7):
    def forward(self, x):
        x0 = self.conv0(x)
        x = self.max_1(x0)
        x = self.conv2(x)
        x = self.max_3(x)
        x = self.conv4_6(x)
        featuremap = self.max_7(x)
        weight, selected_featuremap = self.channel_select(featuremap)
        y = self.deconv4_6(selected_featuremap)
        y = self.deconv2(y)
        y = self.deconv0(y)
        return x0, featuremap, weight, selected_featuremap, y


class Mirror(nn.Module):
    num_channel = 128

    def __init__(self, channel=None, weights_file=False):
        super(Mirror, self).__init__()
        darknet19 = de_Darknet19()
        seblock = SELayer(self.num_channel)
        # self.deconv = nn.Sequential(darknet19.layer1,darknet19.layer2,darknet19.layer3,)
        # self.reconstruct = nn.Sequential(darknet19.layer4)
        # self.unpool_7 = nn.MaxUnpool2d(2, stride=2)
        self.channel_select = seblock
        self.deconv4_6 = darknet19.layer2
        # self.unpool_3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2 = darknet19.layer3
        # self.unpool_1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv0 = darknet19.layer4

    def forward(self, x):
        weight, y = self.channel_select(x)
        y = self.deconv4_6(y)
        y = self.deconv2(y)
        y = self.deconv0(y)
        return y


if __name__ == '__main__':
    model = HalfMirror()
    im = np.random.randn(1, 512, 26, 26)
    im_variable = Variable(torch.from_numpy(im)).float()
    out = model(im_variable)


