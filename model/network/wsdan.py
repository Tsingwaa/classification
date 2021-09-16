"""
WS-DAN models

Hu et al.,
"See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification",
arXiv:1901.09891

Created: May 04,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

from common.backbone.builder import build_backbone

__all__ = ['WSDAN']
EPSILON = 1e-12

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=0.001)
        if m.bias is not None:
            init.constant(m.bias, 0)

# Bilinear Attention Pooling
class WSDAN_AP(nn.Module):
    def __init__(self):
        super(WSDAN_AP, self).__init__()

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))
        phi_I = torch.einsum('imjk,injk->imn', attentions, features)
        phi_I = phi_I / float(H * W)
        phi_I = torch.sign(phi_I) * torch.sqrt(torch.abs(phi_I) + EPSILON)

        phi_attention = phi_I / torch.sum(phi_I + 1e-12, dim=1, keepdim=True)
        phi_feature = phi_I / torch.sum(phi_I + 1e-12, dim=2, keepdim=True)
        phi_I = phi_attention * phi_feature

        pooling_features = torch.sum(phi_I, dim=1)
        pooling_features = F.normalize(pooling_features, dim=-1)

        return pooling_features


## Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self):
        super(BAP, self).__init__()

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()
        
        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix

# Bilinear Attention Pooling
class BAPOnnx(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAPOnnx, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            #feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)

            new_feature_matrix = torch.matmul(attentions.reshape(attentions.shape[0], attentions.shape[1], attentions.shape[2]*attentions.shape[3]), features.reshape(features.shape[0], features.shape[1], features.shape[2]*features.shape[3]).transpose(2, 1))
            feature_matrix = (new_feature_matrix / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        ones_data = torch.ones(feature_matrix.shape)
        negativeones_data = ones_data - 2
        feature_matrix_sign = torch.where(feature_matrix >= 0, feature_matrix, negativeones_data)
        feature_matrix_sign = torch.where(feature_matrix_sign <= 0, feature_matrix_sign, ones_data)
        feature_matrix = feature_matrix_sign * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        #feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix


# WS-DAN: Weakly Supervised Data Augmentation Network for FGVC
class WSDAN(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e',
            pretrained=False,cls_num_list=None, deploy=False):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

        # Network Initialization
        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'vgg' in net:
            self.features = getattr(vgg, net)(pretrained=pretrained).get_features()
            self.num_features = 512
        elif 'efficientnet' in net:
            self.features = EfficientNet.from_pretrained(net)
            self.features.set_swish(memory_efficient=False)
            self.num_features = self.features._out_channels
        elif 'mobilenet' in net:
            self.features = geffnet.create_model(model_name=net, pretrained=pretrained)
            self.num_features = self.features.conv_head.out_channels
        elif 'resnest269' in net:
            self.features = resnest269(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'resnest200' in net:
            self.features = resnest200(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'resnest101' in net:
            self.features = resnest101(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'se_resnet50' in net:
            self.features = se_resnet50().get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'se_resnext50_32x4d' in net:
            self.features = se_resnext50_32x4d().get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported net: %s' % net)

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.deploy = deploy #True
        #'''
        # Attention Maps
        if self.net != 'inception_mixed_7c':
            self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAPOnnx()
        # self.bap = WSDAN_AP()

        # Classification Layer
        #self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)
        if not self.deploy:
            self.fc = nn.Linear(self.num_features, self.num_classes, bias=False)
            self.fc.apply(weights_init)
        #'''

        # ArcMargin
        margin = 'arc' # arc lda
        if 'arc' in margin:
            self.margin = ArcMarginProduct(self.num_features, self.num_classes)
        elif 'lda' in margin:
            self.margin = LDAMargin(cls_num_list=cls_num_list, in_feature=self.num_features, out_feature=self.num_classes)

        logging.info('WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes, self.M))

    def forward(self, x, y=None, get_fea=False):
        #batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if False:#get_fea:
            feature_matrix = self.pool(feature_maps)
            feature_matrix = feature_matrix.view(feature_matrix.size()[0], -1)
            #print(feature_matrix.size())
            if y is not None:#self.training:
                arc = self.margin(feature_matrix, y)
                return feature_matrix, arc# attention_map
            return feature_matrix

        #'''
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]
        attention_maps = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W) # extra
        feature_matrix = self.bap(feature_maps, attention_maps)

        # Classification
        if self.deploy:
            return feature_matrix
        p = self.fc(feature_matrix * 100.)

        # p: (B, self.num_classes)
        # feature_matrix: (B, M * C)
        # attention_map: (B, 2, H, W) in training, (B, 1, H, W) in val/testing
        if y is not None:#self.training:
            arc = self.margin(feature_matrix, y)
            return p, feature_matrix, attention_maps, arc# attention_map
        return p, feature_matrix, attention_maps# attention_map
        #'''

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN, self).load_state_dict(model_dict)
