# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from util.network import WeightLoader


def conv_bn_leaky(in_channels, out_channels, kernel_size, return_module=False):
    padding = int((kernel_size - 1) / 2)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)]
    # layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
    #                     stride=1, padding=padding, bias=False),
    #           nn.LeakyReLU(0.1, inplace=True)]
    if return_module:
        return nn.Sequential(*layers)
    else:
        return layers


class de_Darknet19(nn.Module):

    cfg = {
        #'layer0': [256, 512, 256, 512, 256, 'M'],
        #'layer1': [128, 256, 128, 'M'],
        # 'layer1': ['M'],
        # 'layer2': [64, 128, 64, 'M'],
        # 'layer3': [32, 'M'],
        # 'layer4': [3]

        'layer2': ['M', 64, 128, 64],
        'layer3': ['M', 32],
        'layer4': ['M', 3]
    }

    def __init__(self):
        super(de_Darknet19, self).__init__()
        self.in_channels = 128

        #self.layer0 = self._make_layers(self.cfg['layer0'])
        #self.layer1 = self._make_layers(self.cfg['layer1'])
        self.layer2 = self._make_layers(self.cfg['layer2'])
        self.layer3 = self._make_layers(self.cfg['layer3'])
        self.layer4 = self._make_layers(self.cfg['layer4'])

    def forward(self, x):
        #x = self.layer0(x)
        #x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _make_layers(self, layer_cfg):
        layers = []

        # set the kernel size of the first conv block = 3
        kernel_size = 3
        for v in layer_cfg:
            if v == 'M':
                #layers += [nn.UpsamplingNearest2d(scale_factor=1)]
                #layers += [nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True)]
                layers += [nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.in_channels,  kernel_size=4, stride=2, padding=1, output_padding=0, bias= False)]
                #layers += [nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True)]
                #layers += [nn.PixelShuffle(1)]
            elif v == 'S':
                layers += [nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True)]

            else:
                layers += conv_bn_leaky(self.in_channels, v, kernel_size)
                kernel_size = 1 if kernel_size == 3 else 3
                self.in_channels = v
        return nn.Sequential(*layers)

    # very ugly code !! need to reconstruct
    def load_weights(self, weights_file):
        weights_loader = WeightLoader()
        weights_loader.load(self, weights_file)

#
# if __name__ == '__main__':
#     im = np.random.randn(1, 3, 224, 224)
#     im_variable = Variable(torch.from_numpy(im)).float()
#     model = de_Darknet19()
#     out = model(im_variable)
#     print(out.size())
#     print(model)
