"""
src: https://github.com/gsp-27/pytorch_Squeezenet/blob/master/model.py
"""

import torch
#===== 
# import torch.nn as nn
# import torch.nn.functional as F 
# from torch.autograd import Variable
import ext3.nn as nn
#=====
import numpy as np
import torch.optim as optim
import math

#=====
# EDITS.
# - nn.ReLU(True) ---> nn.ReLU().
# - torch.cat ---> nn.Cat().
# - torch.flatten ---> no edit.
#=====

class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU()

        #===== 
        # missing layers.
        self.cat = nn.Cat(dim=1) 
        #=====

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        # out = torch.cat([out1, out2], 1)
        out = self.cat(out1, out2)
        out = self.relu2(out)
        return out


class SqueezeNet(nn.Module):
    #===== 
    # def __init__(self):
    def __init__(self, width_mult=1.0, num_classes=10):
    #=====
        super(SqueezeNet, self).__init__()
        
        #===== 
        # handle width_mult.
        self.cfg = configs[width_mult]
        #=====

        #===== 
        # self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
        # self.bn1 = nn.BatchNorm2d(96)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        # self.fire2 = fire(96, 16, 64)
        # self.fire3 = fire(128, 16, 64)
        # self.fire4 = fire(128, 32, 128)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        # self.fire5 = fire(256, 32, 128)
        # self.fire6 = fire(256, 48, 192)
        # self.fire7 = fire(384, 48, 192)
        # self.fire8 = fire(384, 64, 256)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        # self.fire9 = fire(512, 64, 256)
        # self.conv2 = nn.Conv2d(512, 10, kernel_size=1, stride=1)
        # self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)

        # handle width_mult.
        # - first layers.
        self.conv1 = nn.Conv2d(3, self.cfg[0][0], kernel_size=3, stride=1, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(self.cfg[0][0])
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16

        # - middle layers.
        self.fire2 = fire(*self.cfg[0])
        self.fire3 = fire(*self.cfg[1])
        self.fire4 = fire(*self.cfg[2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire5 = fire(*self.cfg[3])
        self.fire6 = fire(*self.cfg[4])
        self.fire7 = fire(*self.cfg[5])
        self.fire8 = fire(*self.cfg[6])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire9 = fire(*self.cfg[7])

        # - last layers.
        self.conv2 = nn.Conv2d(2*self.cfg[7][-1], num_classes, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        #=====

        #===== 
        # missing layers.
        self.input = nn.Input()
        # self.softmax = nn.LogSoftmax(dim=1)
        #=====
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        #===== 
        x = self.input(x)
        #=====
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        #===== 
        # x = self.softmax(x)
        # x = torch.flatten(x, 1)
        x = nn.flatten(x, 1)
        #=====
        return x

def fire_layer(inp, s, e):
    f = fire(inp, s, e)
    return f

#===== 
# handle width_mult.
configs = {
    # (input_channel, squeeze_channel, expand_channel)
    0.10: [
        ( 12,  4,   8),
        ( 16,  4,   8),
        ( 16,  4,  12),
        ( 24,  4,  12),
        ( 24,  8,  20),
        ( 40,  8,  20),
        ( 40,  8,  24),
        ( 48,  8, 256),
    ],
    0.15: [
        ( 16,  4,  12),
        ( 24,  4,  12),
        ( 24,  8,  20),
        ( 40,  8,  20),
        ( 40,  8,  28),
        ( 56,  8,  28),
        ( 56, 12,  40),
        ( 80, 12, 256),
    ],
    0.20: [
        ( 20,  4,  12),
        ( 24,  4,  12),
        ( 24,  8,  24),
        ( 48,  8,  24),
        ( 48, 12,  40),
        ( 80, 12,  40),
        ( 80, 12,  52),
        (104, 12, 256),
    ],
    0.25: [
        ( 24,  4,  16),
        ( 32,  4,  16),
        ( 32,  8,  32),
        ( 64,  8,  32),
        ( 64, 12,  48),
        ( 96, 12,  48),
        ( 96, 16,  64),
        (128, 16, 256),
    ],
    0.50: [
        ( 48,  8,  32),
        ( 64,  8,  32),
        ( 64, 16,  64),
        (128, 16,  64),
        (128, 24,  96),
        (192, 24,  96),
        (192, 32, 128),
        (256, 32, 256),
    ],
    1.0: [
        ( 96, 16,  64),
        (128, 16,  64),
        (128, 32, 128),
        (256, 32, 128),
        (256, 48, 192),
        (384, 48, 192),
        (384, 64, 256),
        (512, 64, 256),
    ],
}
#=====

#===== 
# def squeezenet(pretrained=False):
#     net = SqueezeNet()
#     # inp = Variable(torch.randn(64,3,32,32))
#     # out = net.forward(inp)
#     # print(out.size())
#     return net
#=====

# if __name__ == '__main__':
#     squeezenet()
