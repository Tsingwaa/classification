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
from model.module.clustering_affinity import ClusteringAffinity
from model.network.builder import Networks
from torch.nn import Parameter

__all__ = [
    'ResNet_CIFAR', 'ResNet20_CIFAR', 'ResNet32_CIFAR', 'ResNet56_CIFAR',
    'ResNet110_CIFAR'
]


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

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes,
                              self.expansion * planes,
                              kernel_size=1,
                              stride=stride,
                              bias=False),
                    nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet_CIFAR(nn.Module):

    def __init__(self,
                 block,
                 num_blocks,
                 num_classes=10,
                 use_norm=False,
                 **kwargs):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3,
                               16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if use_norm:
            self.fc = NormedLinear(64, num_classes)
        else:
            self.fc = nn.Linear(64, num_classes)

        self.fc_2 = nn.Linear(64, 2)
        self.fc_N = nn.Linear(2, num_classes)
        # Init ClusteringAffinity class object
        # self.bn_a = nn.BatchNorm1d(64)
        self.affinity = ClusteringAffinity(num_classes=num_classes,
                                           num_centers=kwargs.get(
                                               'num_centers', 5),
                                           sigma=kwargs.get('sigma', 10),
                                           feat_dim=64,
                                           init_weight=True)

        if kwargs.get("proj_head", False):  # BN前的linear层取消bias
            self.proj_head = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64, bias=False),
                nn.BatchNorm1d(64),
            )
        if kwargs.get("pred_head", False):
            self.pred_head = nn.Sequential(
                nn.Linear(64, 32, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 64),
            )

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x1, x2=None, out_type="fc"):
        if 'simsiam' in out_type:
            x1 = self.extract(x1)
            x2 = self.extract(x2)
            z1 = self.proj_head(x1)
            z2 = self.proj_head(x2)

            p1 = self.pred_head(z1)
            p2 = self.pred_head(z2)
            if out_type == "simsiam+fc":
                fc1 = self.fc(x1)
                fc2 = self.fc(x2)
                return p1, p2, z1.detach(), z2.detach(), fc1, fc2
            return p1, p2, z1.detach(), z2.detach()

        elif out_type in ["supcon", "simclr"]:
            x1 = self.extract(x1)
            logits = self.fc(x1)
            norm_vec = F.normalize(self.pred_head(x1), dim=1)
            return logits, norm_vec

        elif out_type == "pred_head":
            x = self.extract(x1)
            x = self.pred_head(x)
            return F.normalize(x, dim=1)

        else:
            x1 = self.extract(x1)
            if 'fc' in out_type:
                return self.fc(x1)
            elif "vec" in out_type:
                return x1
            else:
                raise TypeError

    def extract(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        feat_vec = torch.flatten(x, 1)  # (N, 64)

        return feat_vec


class ResNet_CIFAR2(nn.Module):

    def __init__(self,
                 block,
                 num_blocks,
                 num_classes=10,
                 use_norm=False,
                 **kwargs):
        super(ResNet_CIFAR2, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3,
                               16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, out_type='fc'):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)

        feat_map = self.layer3[:-1](x)

        if out_type == 'map':
            return feat_map
        else:
            feat_map = self.layer3[-1](feat_map)
            feat_vec = self.avgpool(feat_map)
            feat_vec = torch.flatten(feat_vec, 1)

            if out_type == 'vec':
                return feat_vec
            elif out_type == 'mlp':
                return self.mlp(feat_vec)
            else:
                return self.fc(feat_vec)


@Networks.register_module("ResNet20_CIFAR")
class ResNet20_CIFAR(ResNet_CIFAR):

    def __init__(self,
                 num_classes,
                 num_blocks=[3, 3, 3],
                 use_norm=False,
                 **kwargs):
        super(ResNet20_CIFAR, self).__init__(block=BasicBlock,
                                             num_classes=num_classes,
                                             num_blocks=num_blocks,
                                             **kwargs)


@Networks.register_module("ResNet32_CIFAR")
class ResNet32_CIFAR(ResNet_CIFAR):

    def __init__(self,
                 num_classes,
                 num_blocks=[5, 5, 5],
                 use_norm=False,
                 **kwargs):
        super(ResNet32_CIFAR, self).__init__(block=BasicBlock,
                                             num_classes=num_classes,
                                             num_blocks=num_blocks,
                                             **kwargs)


@Networks.register_module("ResNet32_CIFAR2")
class ResNet32_CIFAR2(ResNet_CIFAR2):

    def __init__(self,
                 num_classes,
                 num_blocks=[5, 5, 5],
                 use_norm=False,
                 **kwargs):
        super(ResNet32_CIFAR2, self).__init__(block=BasicBlock,
                                              num_classes=num_classes,
                                              num_blocks=num_blocks,
                                              **kwargs)


@Networks.register_module("ResNet44_CIFAR")
class ResNet44_CIFAR(ResNet_CIFAR):

    def __init__(self,
                 num_classes,
                 num_blocks=[7, 7, 7],
                 use_norm=False,
                 **kwargs):
        super(ResNet44_CIFAR, self).__init__(block=BasicBlock,
                                             num_classes=num_classes,
                                             num_blocks=num_blocks,
                                             **kwargs)


@Networks.register_module("ResNet56_CIFAR")
class ResNet56_CIFAR(ResNet_CIFAR):

    def __init__(self,
                 num_classes,
                 num_blocks=[9, 9, 9],
                 use_norm=False,
                 **kwargs):
        super(ResNet56_CIFAR, self).__init__(block=BasicBlock,
                                             num_classes=num_classes,
                                             num_blocks=num_blocks,
                                             **kwargs)


@Networks.register_module("ResNet110_CIFAR")
class ResNet110_CIFAR(ResNet_CIFAR):

    def __init__(self,
                 num_classes,
                 num_blocks=[18, 18, 18],
                 use_norm=False,
                 **kwargs):
        super(ResNet110_CIFAR, self).__init__(block=BasicBlock,
                                              num_classes=num_classes,
                                              num_blocks=num_blocks,
                                              **kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params:", total_params)
    print(
        "Total layers:",
        len(
            list(
                filter(lambda p: p.requires_grad and len(p.data.size()) > 1,
                       net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
