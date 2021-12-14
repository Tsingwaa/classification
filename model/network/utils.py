"""Utility for Network"""
from torch import nn


def freeze_model(model, unfreeze_keys=['fc']):
    for k, v in model.named_parameters():
        if any(key in k for key in unfreeze_keys):
            v.required_grad = True
        else:
            v.required_grad = False

    return model


def init_linear_weight(model):
    for m in model.modules():
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, gain=1)
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
