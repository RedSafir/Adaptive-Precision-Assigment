'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
#===== 
# import torch.nn as nn
# import torch.nn.functional as F
import ext3.nn as nn
from   ext3.util import _make_divisible
import torch.nn.init as init
#=====

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        #===== 
        # missing layers.
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if self.stride==1: self.add = nn.Add()
        #=====

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.relu1(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out = out + self.shortcut(x) if self.stride==1 else out
        out = self.add(out, self.shortcut(x)) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    #===== 
    # def __init__(self, num_classes=10):
    def __init__(self, width_mult=1.0, num_classes=10):
    #=====
        super(MobileNetV2, self).__init__()

        #===== 
        # handle width_mult. (src: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
        self.width_mult    = width_mult
        self.round_nearest = 4 if width_mult <= 0.25 else 8
        input_channel = 32
        last2_channel = self.cfg[-1][1] # second last.
        last_channel  = 1280
        #=====

        #===== 
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False) # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        # self.bn1 = nn.BatchNorm2d(32)
        # self.layers = self._make_layers(in_planes=32)
        # self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn2 = nn.BatchNorm2d(1280)
        # self.linear = nn.Linear(1280, num_classes)

        # handle width_mult. (src: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
        # - first layers.
        input_channel = _make_divisible(input_channel * self.width_mult, self.round_nearest)
        self.conv1    = nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False) # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.bn1      = nn.BatchNorm2d(input_channel)
        # - middle layers.
        self.layers   = self._make_layers(in_planes=input_channel)
        # - last layers.
        input_channel = _make_divisible(last2_channel * self.width_mult,           self.round_nearest)
        last_channel  = _make_divisible(last_channel  * max(1.0, self.width_mult), self.round_nearest)
        self.conv2    = nn.Conv2d(input_channel, last_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2      = nn.BatchNorm2d(last_channel)
        # - linear layer.
        self.linear   = nn.Linear(last_channel, num_classes)
        #=====
        
        #===== 
        # missing layers.
        self.input = nn.Input()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.avgpool2d = nn.AvgPool2d(4)
        #=====
        
        #===== 
        # weight initialization. (src: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.zeros_(m.bias)
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.zeros_(m.bias)
        #=====

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            #===== 
            # handle width_mult. (src: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
            out_planes = _make_divisible(out_planes * self.width_mult, self.round_nearest)
            #=====
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        #===== 
        x = self.input(x)
        #=====
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        # out = F.relu(self.bn2(self.conv2(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        # out = F.avg_pool2d(out, 4) # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = self.avgpool2d(out)
        # out = out.view(out.size(0), -1)
        out = nn.view(out, (out.size(0), -1))
        out = self.linear(out)
        return out

    
def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
