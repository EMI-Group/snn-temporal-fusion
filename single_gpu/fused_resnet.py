"""
This module contains the classes `BasicBlock`, `Bottleneck`, and `SpikingResNet`, which are inspired by the implementation 
in SpikingJelly. This refers to the following publication:

Fang, W., Chen, Y., Ding, J., Yu, Z., Masquelier, T., Chen, D., Huang, L., Zhou, H., Li, G., Tian, Y.
SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence.
Science Advances, vol. 9, no. 40, eadi1480, 2023.

We acknowledge and thank the contributors of SpikingJelly for their work.
"""


import torch
import torch.nn as nn
import neurons


def MergeDimension(x):
    return x.view(-1, *x.shape[2:])


def SplitDimension(x, time_step):
    return x.view(time_step, x.shape[0]//time_step, *x.shape[1:])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, time_step, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.sn1 = neurons.LIF(decay=0.2, threshold=0.3, time_step=time_step)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.sn2 = neurons.LIF(decay=0.2, threshold=0.3, time_step=time_step)
        self.downsample = downsample
        self.time_step = time_step

    def forward(self, x):
        x = MergeDimension(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = SplitDimension(x, self.time_step)

        x = self.sn1(x)

        x = MergeDimension(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = SplitDimension(x, self.time_step)
        
        x = self.sn2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, time_step, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        width = planes
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.sn1 = neurons.LIF(decay=0.2, threshold=0.3, time_step=time_step)
        self.conv2 = nn.Conv2d(width, width, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(width)
        self.sn2 = neurons.LIF(decay=0.2, threshold=0.3, time_step=time_step)
        self.conv3 = nn.Conv2d(width, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*self.expansion)
        self.sn3 = neurons.LIF(decay=0.2, threshold=0.3, time_step=time_step)
        self.downsample = downsample
        self.stride = stride
        self.time_step = time_step

    def forward(self, x):
        x = MergeDimension(x)
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = SplitDimension(x, self.time_step)

        x = self.sn1(x)

        x = MergeDimension(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = SplitDimension(x, self.time_step)

        x = self.sn2(x)

        x = MergeDimension(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += identity
        x = SplitDimension(x, self.time_step)

        x = self.sn3(x)

        return x


class SpikingResNet(nn.Module):
    def __init__(self, block, layers, time_step, num_classes):
        super(SpikingResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.time_step = time_step
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = neurons.LIF(decay=0.2, threshold=0.3, time_step=time_step)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, stride=stride, kernel_size=1, bias=False),
                norm_layer(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.time_step, stride, downsample, norm_layer))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.time_step, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = MergeDimension(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = SplitDimension(x, self.time_step)

        x = self.sn1(x)

        x = MergeDimension(x)
        x = self.maxpool(x)
        x = SplitDimension(x, self.time_step)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = MergeDimension(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = SplitDimension(x, self.time_step)

        return x


def spiking_resnet18(time_step, num_classes):
    return SpikingResNet(block=BasicBlock, layers=[2, 2, 2, 2], time_step=time_step, num_classes=num_classes)


def spiking_resnet34(time_step, num_classes):
    return SpikingResNet(block=BasicBlock, layers=[3, 4, 6, 3], time_step=time_step, num_classes=num_classes)


def spiking_resnet50(time_step, num_classes):
    return SpikingResNet(block=Bottleneck, layers=[3, 4, 6, 3], time_step=time_step, num_classes=num_classes)
