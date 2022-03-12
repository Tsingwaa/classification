# import math

import torch
import torch.nn as nn
# import torch.nn.functional as F
from model.network.builder import Networks

model_urls = {
    "resnet18":
    "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth",
    "resnet34":
    "https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth",
    "resnet50":
    "https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth",
    "resnet101":
    "https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth",
    "resnet152":
    "https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))

        out = self.relu2(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = out + residual
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BBN_ResNet(nn.Module):

    def __init__(self,
                 block_type,
                 num_blocks,
                 last_layer_stride=2,
                 num_classes=1000,
                 pretrained=False,
                 pretrained_fpath=""):
        super(BBN_ResNet, self).__init__()
        self.inplanes = 64
        self.block = block_type

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(num_blocks[1], 128, stride=2)
        self.layer3 = self._make_layer(num_blocks[2], 256, stride=2)
        self.layer4 = self._make_layer(num_blocks[3] - 1,
                                       512,
                                       stride=last_layer_stride)

        self.cb_block = self.block(self.inplanes, self.inplanes // 4, stride=1)
        self.rb_block = self.block(self.inplanes, self.inplanes // 4, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * block_type.expansion, num_classes)

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict[
            "state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "fc" not in k and "classifier" not in k:
                k = k.replace("backbone.", "")
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for now_stride in strides:
            layers.append(self.block(self.inplanes, planes, stride=now_stride))
            self.inplanes = planes * self.block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):

        if "classifier_flag" in kwargs:
            out = self.avgpool(x)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if "feature_cb" in kwargs:
            out = self.cb_block(out)
            return out
        elif "feature_rb" in kwargs:
            out = self.rb_block(out)
            return out
        out1 = self.cb_block(out)
        out2 = self.rb_block(out)
        out = torch.cat((out1, out2), dim=1)

        return out


@Networks.register_module('ResNet50_BBN')
class ResNet50_BBN(BBN_ResNet):

    def __init__(self, num_classes, **kwargs):
        super(ResNet50_BBN, self).__init__(block_type=BottleNeck,
                                           num_blocks=[3, 4, 6, 3],
                                           last_layer_stride=2,
                                           num_classes=num_classes,
                                           **kwargs)
