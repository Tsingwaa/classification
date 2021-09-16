#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: ArcMarginProduct.py
@time: 2018/12/25 9:13
@desc: additive angular margin for arcface/insightface
'''

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np


class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575,
                 s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing
        # while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if label is None:
            return cosine
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine.float() - self.th) > 0,
                              phi.float(), cosine.float() - self.mm)

        # one_hot = torch.zeros(cosine.size(), device='cuda'
        # if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


class LDAMargin(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, class_weight=None, s=32,
                 in_feature=128, out_feature=256):
        super(LDAMargin, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        # self.class_weight = class_weight

        self.weight = Parameter(torch.Tensor(in_feature, out_feature))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, target):
        x = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))

        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(
            self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        output = output * self.s
        return output
