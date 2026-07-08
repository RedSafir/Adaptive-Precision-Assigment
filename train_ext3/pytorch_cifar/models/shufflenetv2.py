'''ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''
import torch
#===== 
# import torch.nn as nn
# import torch.nn.functional as F 
import ext3.nn as nn
import numpy as np
#=====


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        # return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)
        x = nn.view   (x, (N, g, C//g, H, W))
        x = nn.permute(x, (0, 2, 1, 3, 4))
        x = nn.reshape(x, (N, C, H, W))
        return x


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

        #===== 
        # missing layers.
        assert(ratio == 0.5)
        self.split1 = nn.SplitHalf(dim=1, index=0) # <==> x[:, :c, :, :].
        self.split2 = nn.SplitHalf(dim=1, index=1) # <==> x[:, c:, :, :].
        #=====

    def forward(self, x):
        # # c = int(x.size(1) * self.ratio)
        # # return x[:, :c, :, :], x[:, c:, :, :]
        # c = int(x.size(1) * self.ratio)
        # return nn.getitem(x, np.s_[:, :c, :, :]), nn.getitem(x, np.s_[:, c:, :, :])
        return self.split1(x), self.split2(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()
        
        #===== 
        # missing layers.
        self.relu1 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.cat = nn.Cat(dim=1)
        #=====

    def forward(self, x):
        x1, x2 = self.split(x)
        # out = F.relu(self.bn1(self.conv1(x2)))
        out = self.relu1(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        # out = F.relu(self.bn3(self.conv3(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        # out = torch.cat([x1, out], 1)
        out = self.cat(x1, out)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # right
        self.conv3 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.shuffle = ShuffleBlock()

        #===== 
        # missing layers.
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.cat = nn.Cat(dim=1)
        #=====

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        # out1 = F.relu(self.bn2(self.conv2(out1)))
        out1 = self.relu2(self.bn2(self.conv2(out1)))
        # right
        # out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.relu3(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        # out2 = F.relu(self.bn5(self.conv5(out2)))
        out2 = self.relu5(self.bn5(self.conv5(out2)))
        # concat
        # out = torch.cat([out1, out2], 1)
        out = self.cat(out1, out2)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(nn.Module):
    #===== 
    # def __init__(self, net_size):
    def __init__(self, net_size=1.0, num_classes=10): # net_size = width_mult.
    #=====
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv2d(out_channels[2], out_channels[3],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        #===== 
        # self.linear = nn.Linear(out_channels[3], 10)
        self.linear = nn.Linear(out_channels[3], num_classes)
        #=====

        #===== 
        # missing layers.
        self.input = nn.Input()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.avgpool2d = nn.AvgPool2d(4)
        #=====

        #===== 
        # no init in torchvision: https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
        #=====

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        #===== 
        x = self.input(x)
        #=====
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.relu1(self.bn1(self.conv1(x)))
        # # out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.relu(self.bn2(self.conv2(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool2d(out)
        # out = out.view(out.size(0), -1)
        out = nn.view(out, (out.size(0), -1))
        out = self.linear(out)
        return out


configs = {
    #===== 
    # support more width_mult.
    0.10: {
        'out_channels': (12, 24, 48, 1024),
        'num_blocks': (3, 7, 3)
    },
    0.15: {
        'out_channels': (16, 32, 64, 1024),
        'num_blocks': (3, 7, 3)
    },
    0.20: {
        'out_channels': (24, 48, 96, 1024),
        'num_blocks': (3, 7, 3)
    },
    0.25: {
        'out_channels': (28, 56, 112, 1024),
        'num_blocks': (3, 7, 3)
    },
    0.50: {
        # 'out_channels': (48, 96, 192, 1024),
        'out_channels': (56, 112, 224, 1024),
        'num_blocks': (3, 7, 3)
    },
    0.75: {
        'out_channels': (88, 176, 352, 1024),
        'num_blocks': (3, 7, 3)
    },
    #=====
    1.0: {
        'out_channels': (116, 232, 464, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2.0: {
        'out_channels': (244, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}


def test():
    net = ShuffleNetV2(net_size=0.5)
    x = torch.randn(3, 3, 32, 32)
    y = net(x)
    print(y.shape)


# test()
