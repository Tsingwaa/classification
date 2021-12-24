"""Utility for Network"""
import torch
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


class Normalization(nn.Module):
    def __init__(self, mean, std, n_channels=3):
        super(Normalization, self).__init__()
        self.n_channels = n_channels
        if mean is None:
            mean = [.0] * n_channels
        if std is None:
            std = [.1] * n_channels
        self.mean = torch.tensor(list(mean)).reshape(
            (1, self.n_channels, 1, 1))
        self.std = torch.tensor(list(std)).reshape((1, self.n_channels, 1, 1))
        self.mean = nn.Parameter(self.mean, requires_grad=False)
        self.std = nn.Parameter(self.std, requires_grad=False)

    def forward(self, x):
        y = (x - self.mean / self.std)
        return y


class MixBatchNorm2d(nn.BatchNorm2d):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].

    If you set batch_type as 'mix', this network will using one batchnorm
    (main bn) to calculate the features corresponding to [:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features
    of [N:, 3, 224, 224].

    During training, the batch_type should be set as 'mix'.

    During validation, we only need the results of the features using some
    specific batchnormalization.
    if you set batch_type as 'clean', the features are calculated using main
    bn;
    if you set it as 'adv', the features are calculated using auxiliary bn.

    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set
    the batch_type recursively. It should be noticed that the batch_type
    should be set as 'adv' while attacking.
    '''

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(num_features, eps, momentum,
                                             affine, track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features,
                                     eps=eps,
                                     momentum=momentum,
                                     affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, x):
        if self.batch_type == 'adv':
            output = self.aux_bn(x)
        elif self.batch_type == 'clean':
            output = super(MixBatchNorm2d, self).forward(x)
        else:
            assert self.batch_type == 'mix'
            clean_x, adv_x = x.chunk(2, 0)  # 沿0维二等分
            clean_output = super(MixBatchNorm2d, self).forward(clean_x)
            adv_output = self.aux_bn(adv_x)
            output = torch.cat((clean_output, adv_output), 0)  # 沿0维合并
        return output
