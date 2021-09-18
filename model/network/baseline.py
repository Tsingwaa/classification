"""
WS-DAN models, Hu et al., arXiv:1901.09891
"See Better Before Looking Closer: Weakly Supervised Data Augmentation Network
for Fine-Grained Visual Classification",

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
from common.network.builder import Networks
from common.network.layers import ArcMarginProduct
__all__ = ['BASELINE']
EPSILON = 1e-12


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


# Bilinear Attention Pooling
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
        feature_matrix = (torch.einsum('imjk,injk->imn',
                          (attentions, features)) / float(H * W)).view(B, -1)

        # sign-sqrt
        feature_matrix = torch.sign(
            feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix


@Networks.register_module("baseline")
class BASELINE(nn.Module):
    def __init__(self, config=None, **kwargs):
        super(BASELINE, self).__init__()
        try:
            self.num_classes = kwargs['num_classes']
        except Exception as Error:
            print(Error)

        pretrained = kwargs['pretrained']

        backbone_config = config['backbone']
        if backbone_config is None:
            raise ValueError("Param can't be None")

        backbone_name = backbone_config.get("name", None)
        backbone_param = backbone_config.get("param", None)
        self.features = build_backbone(backbone_name, **backbone_param)
        if "resnet" in backbone_name:
            if backbone_name == "resnet18" or backbone_name == "resnet34":
                expansion = 1
            else:
                expansion = 4
            self.num_features = 512 * expansion

        self.gap = nn.AdaptiveAvgPool2d(1)
        # Classification Layer

        apex = True
        if apex:
            self.bn_neck = nn.BatchNorm1d(self.num_features)
            self.bn_neck.bias.requires_grad_(False)  # no shift
        else:
            self.bn_neck = nn.BatchNorm1d(
                self.num_features, momentum=0.1, affine=False)

        self.bn_neck.apply(weights_init_kaiming)
        self.embedding_in_planes = 512
        self.embedding = nn.Linear(
            self.num_features, self.embedding_in_planes, bias=True)
        self.embedding.apply(weights_init_kaiming_embedding)

        has_embedding = True

        network_param = config['param']
        self.has_coslayer = network_param.get("use_arc", False)
        self.deploy = network_param.get("deploy", False)

        if self.has_coslayer:
            scale = network_param.get("scale", 30)
            margin = network_param.get("margin", 0.45)

        if self.has_coslayer:
            if has_embedding:
                self.cos_layer = ArcMarginProduct(self.embedding_in_planes,
                                                  self.num_classes, s=scale, m=margin)
            else:
                self.cos_layer = ArcMarginProduct(self.num_features,
                                                  self.num_classes, s=scale, m=margin)
        else:
            if has_embedding:
                self.fc = nn.Linear(self.embedding_in_planes,
                                    self.num_classes, bias=False)
            else:
                self.fc = nn.Linear(self.num_features,
                                    self.num_classes, bias=False)
                self.fc.apply(weights_init)

        logging.info('Baseline: using {} as feature extractor, \
                num_classes: {}'.format(config['backbone']['name'],
                                        self.num_classes))

    def forward(self, x, y=None, get_fea=False):
        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        # gap
        gap_feat = self.gap(feature_maps)
        gap_feat = gap_feat.flatten(1)
        # bn neck
        gap_feat = self.bn_neck(gap_feat)
        # Classification
        embedding = self.embedding(gap_feat)
        if self.training:
            if self.has_coslayer:
                pro = self.cos_layer(embedding, y)
            else:
                pro = self.fc(embedding)
            return pro
        else:
            if not self.deploy:
                pro = self.fc(embedding)
                return pro
            elif self.deploy:
                normed_feat = F.normalize(embedding, dim=1)
                return normed_feat

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' %
                         type(self).__name__)
            not_loaded_keys = [
                k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') %
                         tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(BASELINE, self).load_state_dict(model_dict)
