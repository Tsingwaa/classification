'''
Properly implemented ResNet for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
# from pudb import set_trace
from model.network.builder import Networks
from .utils import Normalization, MixBatchNorm2d

__all__ = ['NormResNet32_CIFAR']


def _weights_init(m):
    # classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=None,
                 option='A'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes//4, planes//4),
                        "constant",
                        0
                    )
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NormResNet_CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10, use_norm=False,
                 mean=None, std=None, dual_BN=False, norm_layer=None,
                 **kwargs):
        super(NormResNet_CIFAR, self).__init__()
        self.in_planes = 16

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.normalize = Normalization(mean, std)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(16)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if use_norm:
            self.fc = NormedLinear(64, num_classes)
        else:
            self.fc = nn.Linear(64, num_classes)

        self.fc_2 = nn.Linear(64, 2)
        self.fc_N = nn.Linear(2, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        norm_layer = self._norm_layer
        stride_list = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in stride_list:
            layers.append(
                block(self.in_planes, planes, stride, norm_layer))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, out='fc'):
        # x: (N, 3, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))  # (N, 16, 32, 32)
        x = self.layer1(x)  # (N, 16, 32, 32)
        x = self.layer2(x)  # (N, 32, 16, 16)
        x = self.layer3(x)  # (N, 64, 8, 8)
        x = self.avgpool(x)  # (N, 64, 1, 1)
        feat = torch.squeeze(x)  # (N, 64)
        if out == 'feat':
            return feat  # (N, 64)
        elif '2' in out:
            feat_2d = F.relu(self.fc_2(x))  # (N, 2)
            if out == 'feat_2d':
                return feat_2d  # (N, 2)
            else:  # out == 'fc_2d_N'
                return self.fc_N(feat_2d)  # (N, C)
        else:
            return self.fc(feat)  # (N, C)


# @Networks.register_module("NormResNet32_CIFAR")
class NormResNet32_CIFAR(NormResNet_CIFAR):
    def __init__(self, num_classes, layers=[5, 5, 5], dual_BN=True,
                 mean=None, std=None, use_norm=False, **kwargs):
        norm_layer = MixBatchNorm2d if dual_BN else None
        super(NormResNet32_CIFAR, self).__init__(
            block=BasicBlock,
            num_classes=num_classes,
            layers=layers,
            mean=mean,
            std=std,
            norm_layer=norm_layer,
            **kwargs,
        )


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params:", total_params)
    print(
        "Total layers:",
        len(list(filter(
            lambda p: p.requires_grad and len(p.data.size()) > 1,
            net.parameters()))))


# if __name__ == "__main__":
    # for net_name in __all__:
    #     if net_name.startswith('resnet'):
    #         print(net_name)
    #         test(globals()[net_name]())
    #         print()
    # norm_resnet = NormResNet32_CIFAR(10, mean=[0,]*3, std=[1,]*3)
    # x = torch.randn(5, 3, 32, 32)
    # print(norm_resnet)
    # y_ = norm_resnet(x)
