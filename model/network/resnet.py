# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from model.network.builder import Networks
# from torchvision import models

# @Networks.register_module("ResNet50")
# class ResNet50(nn.Module):

#     def __init__(self, num_classes, pretrained, **kwargs):
#         super(ResNet50, self).__init__()
#         self.num_classes = num_classes

#         backbone = models.resnet50(pretrained=pretrained)
#         self.features = nn.Sequential(*list(backbone.children())[:-1])
#         self.fc_in = 2048

#         self.fc = nn.Linear(self.fc_in, num_classes)

#         if kwargs.get("pred_head", False):
#             self.pred_head = nn.Sequential(
#                 nn.Linear(self.fc_in, 512, bias=False),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(512, 128),
#             )

#     def extract(self, x):
#         feat_map = self.features(x)
#         feat_vec = torch.flatten(feat_map, 1)
#         return feat_vec

#     def forward(self, x1, x2=None, out_type="fc"):
#         if out_type in ["supcon", "simclr"]:
#             feat_vec = self.extract(x1)
#             logits = self.fc(feat_vec)
#             norm_vec = F.normalize(self.pred_head(feat_vec), dim=1)
#             return logits, norm_vec

#         elif out_type == "pred_head":
#             feat_vec = self.extract(x1)
#             de_feat_vec = self.pred_head(feat_vec)
#             return F.normalize(de_feat_vec, dim=1)

#         else:
#             x1 = self.extract(x1)
#             if 'fc' in out_type:
#                 return self.fc(x1)
#             elif "vec" in out_type:
#                 return x1
#             else:
#                 raise TypeError

import torch
# from pudb import set_trace
from model.network.builder import Networks
from torch import nn
from torch.nn import functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
}


def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_in')

        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming_embedding(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0)
        nn.init.constant_(m.bias, 0.0)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal(m.weight, mode='fan_out')

        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal(m.weight, std=0.001)

        if m.bias is not None:
            nn.init.constant(m.bias, 0)


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

        out_type = self.conv1(x)
        out_type = self.bn1(out_type)
        out_type = self.relu(out_type)

        out_type = self.conv2(out_type)
        out_type = self.bn2(out_type)

        if self.downsample is not None:
            identity = self.downsample(x)

        out_type += identity
        out_type = self.relu(out_type)

        return out_type


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling
    # at 3x3 convolution(self.conv2) while original implementation
    # places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"
    # https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy
    # according to https://ngc.nvidia.com/catalog/model-scripts
    #                       /nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out_type = self.conv1(x)
        out_type = self.bn1(out_type)
        out_type = self.relu(out_type)

        out_type = self.conv2(out_type)
        out_type = self.bn2(out_type)
        out_type = self.relu(out_type)

        out_type = self.conv3(out_type)
        out_type = self.bn3(out_type)

        if self.downsample is not None:
            identity = self.downsample(x)

        out_type += identity
        out_type = self.relu(out_type)

        return out_type


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 **kwargs):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,
                                       planes=64,
                                       blocks=layers[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       planes=128,
                                       blocks=layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       planes=256,
                                       blocks=layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       planes=512,
                                       blocks=layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if kwargs.get("fc_3lp", False):
            self.fc_3lp = nn.Sequential(
                nn.Linear(512 * block.expansion, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes),
            )

        if kwargs.get("proj_head", False):  # BN前的linear层取消bias
            self.proj_head = nn.Sequential(
                nn.Linear(512 * block.expansion,
                          512 * block.expansion,
                          bias=False),
                nn.BatchNorm1d(512 * block.expansion),
                nn.ReLU(inplace=True),
                nn.Linear(512 * block.expansion,
                          512 * block.expansion,
                          bias=False),
                nn.BatchNorm1d(512 * block.expansion),
                nn.ReLU(inplace=True),
                nn.Linear(512 * block.expansion,
                          512 * block.expansion,
                          bias=False),
                nn.BatchNorm1d(512 * block.expansion),
            )
        if kwargs.get("pred_head", False):
            self.pred_head = nn.Sequential(
                nn.Linear(512 * block.expansion, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer)
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

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
            if "fc" in out_type:
                return self.fc(x1)
            elif "vec" in out_type:
                return x1
            else:
                raise TypeError

    def extract(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


@Networks.register_module('ResNet18')
class ResNet18(ResNet):

    def __init__(self, num_classes, **kwargs):
        super(ResNet18, self).__init__(block=BasicBlock,
                                       layers=[2, 2, 2, 2],
                                       num_classes=num_classes,
                                       **kwargs)


@Networks.register_module('ResNet34')
class ResNet34(ResNet):

    def __init__(self, num_classes, **kwargs):
        super(ResNet34, self).__init__(block=BasicBlock,
                                       layers=[3, 4, 6, 3],
                                       num_classes=num_classes,
                                       **kwargs)


@Networks.register_module('ResNet50')
class ResNet50(ResNet):

    def __init__(self, num_classes, **kwargs):
        super(ResNet50, self).__init__(block=Bottleneck,
                                       layers=[3, 4, 6, 3],
                                       num_classes=num_classes,
                                       **kwargs)


@Networks.register_module('ResNet101')
class ResNet101(ResNet):

    def __init__(self, num_classes, **kwargs):
        super(ResNet101, self).__init__(block=Bottleneck,
                                        layers=[3, 4, 23, 3],
                                        num_classes=num_classes,
                                        **kwargs)


@Networks.register_module('ResNet152')
class ResNet152(ResNet):

    def __init__(self, num_classes, **kwargs):
        super(ResNet152, self).__init__(block=Bottleneck,
                                        layers=[3, 8, 36, 3],
                                        num_classes=num_classes,
                                        **kwargs)


@Networks.register_module('ResNeXt50')
class ResNeXt50(ResNet):

    def __init__(self, num_classes, **kwargs):
        super(ResNeXt50, self).__init__(block=Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        num_classes=num_classes,
                                        groups=32,
                                        width_per_group=4,
                                        **kwargs)
