###############################################################################
# Filename: center_loss.py
# Author: Tsingwaa
# Email: zengchh3@gmail.com
# Created Time : 2021-12-19 16:48 Sunday
# Last modified: 2021-12-19 16:49 Sunday
# Reference:
#     Wen et al. A Discriminative Feature Learning Approach for Deep Face
#     Recognition. ECCV 2016.
###############################################################################

from __future__ import absolute_import

import torch
from model.loss.builder import Losses
from pudb import set_trace
from torch import nn
from utils import cos_sim, eu_dist
import numpy as np
import torch.nn.functional as F

@Losses.register_module("CB_CenterLoss")
class CB_CenterLoss(nn.Module):
    def __init__(self,
                 num_classes=10,
                 feat_dim=2,
                 weight=None,
                 alpha=0,
                 alpha_dist='eu',
                 **kwargs):
        """Initialize class centers

        Args:
            num_classes (int): number of classes.
            feature_dim (int): dimension of feature vector.
            alpha (float): weight of constraint for distance between centers.
            kwargs (dict): other args.
        """

        super(CB_CenterLoss, self).__init__()
        self.alpha = alpha
        self.alpha_dist = alpha_dist
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.weights = torch.tensor(weight).cuda()
        self.num_classes = num_classes

    def forward(self, feat_vec, labels):
        """
        Args:
            feat_vecs (Tensor, batch_size * feat_dim): feature vectors
            labels (Tensor or List, batch_size * 1): ground truth labels
        """

        self.centers = self.centers.cuda()
        center = self.centers[labels]
        weights = self.weights[labels]
        dist = (feat_vec - center).pow(2).sum(dim=-1)
        dist = weights * dist
        loss = 0.5 * torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        if self.alpha_dist == 'eu':
            dist_func = eu_dist
        elif self.alpha_dist == 'cos':
            dist_func = cos_sim
        else:
            dist_func = None
            print('No distance function')

        if self.alpha > 0:
            intercenter_dist = dist_func(self.centers, self.centers)
            intercenter_dist = torch.triu(intercenter_dist, diagonal=1)
            # 严格上三角
            dist_num = torch.sum(intercenter_dist.ge(0))
            loss -= self.alpha * torch.sum(intercenter_dist) / dist_num

        return loss


if __name__ == '__main__':
    use_gpu = False
    num_classes = 10
    feat_dim = 2

    features = torch.ones(16, 2)
    targets = torch.ones((16, )).long()
    if use_gpu:
        features = features.cuda()
        targets = targets.cuda()

    center_loss = CenterLoss(num_classes, feat_dim, use_gpu)

    loss = center_loss(features, targets)
    print(loss)
