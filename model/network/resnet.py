"""
WS-DAN models

Hu et al., arXiv:1901.09891, "See Better Before Looking Closer: Weakly
Supervised Data Augmentation Network for Fine-Grained Visual Classification"

Created: May 04,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import logging
import torch.nn as nn
from torch.nn import init
from common.backbone.builder import build_backbone
from common.network.builder import Networks
from common.network.layers import ArcMarginProduct

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


@Networks.register_module("resnet")
class RESNET(nn.Module):
    def __init__(self, num_classes=100, pretrained=False, config=None,
                 deploy=False, **kwargs):
        super(RESNET, self).__init__()
        self.num_classes = num_classes

        backbone_config = config['backbone']
        if backbone_config is None:
            raise ValueError("Param can't be None")

        backbone_name = backbone_config.get("name", None)
        backbone_param = backbone_config.get("param", None)
        self.features = build_backbone(backbone_name, **backbone_param)
        self.num_features = 512 * self.features.layer4[-1].expansion

        self.gap = nn.AdaptiveAvgPool2d(1)
        # Classification Layer

        self.fc = nn.Linear(self.num_features, self.num_classes, bias=False)
        self.fc.apply(weights_init)

        logging.info('resnet : using {} as feature extractor, \
                num_classes: {}'.format(config['backbone']['name'], self.num_classes))

    def forward(self, x, y=None, get_fea=False):
        # Feature Maps
        feature_maps = self.features(x)
        # gap
        gap_feat = self.gap(feature_maps)
        gap_feat = gap_feat.flatten(1)
        # Classification
        pro = self.fc(gap_feat)
        return pro

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
        super(RESNET, self).load_state_dict(model_dict)
