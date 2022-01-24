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
# from pudb import set_trace
from torch import nn

# from utils import cos_sim, eu_dist


@Losses.register_module("CenterLoss")
class CenterLoss(nn.Module):

    def __init__(self,
                 num_classes,
                 feat_dim,
                 weight=None,
                 margin=-1,
                 dist_metric='eu',
                 **kwargs):
        """Initialize class centers

        Args:
            num_classes (int): number of classes.
            feat_dim (int): dimension of feature vector.
            alpha (float): weight of constraint for distance between centers.
            alpha_dist (dist type): optional eu or cos.
            weight (list): whether to weight center loss
            kwargs (dict):
        """

        super(CenterLoss, self).__init__()
        self.weight = weight

        if self.weight is not None:
            self.weight = self.weight.cuda()

        self.margin = margin
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feats, targets):
        """
        Args:
            feat_vecs (Tensor, batch_size * feat_dim): feature vectors
            labels (Tensor or List, batch_size * 1): ground truth labels
        """

        # self.centers = self.centers.cuda()  # (C, d)

        if self.margin == -1:  # 不使用margin
            center = self.centers[targets]  # (N, d)
            dist = (feats - center).pow(2).sum(dim=-1)  # (N, 1)

            if self.weight is not None:
                dist *= self.weight[targets]  # (N, 1)
            loss = 0.5 * torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)
        else:
            feat_expand = feats.unsqueeze(1)  # (N, d) -> (N, 1, d)
            center_expand = self.centers.unsqueeze(0)  # (C, d) -> (1, C, d)

            # (N, 1, d) - (1, C, d) = (N, C, d) --> 2-Norm(N, C)
            dist_all = torch.norm(feat_expand - center_expand, dim=-1)
            targets = targets.view(-1, 1)

            min_dist = torch.min(dist_all, dim=-1)[0]
            loss = torch.clamp(self.margin + dist - min_dist, min=0)

        # if self.alpha_dist == 'eu':
        #     dist_func = eu_dist
        # elif self.alpha_dist == 'cos':
        #     dist_func = cos_sim
        # else:
        #     dist_func = None
        #     print('No distance function')

        # if self.alpha > 0:
        #     intercenter_dist = dist_func(self.centers, self.centers)
        #     intercenter_dist = torch.triu(intercenter_dist, diagonal=1)
        #     # 严格上三角
        #     dist_num = torch.sum(intercenter_dist.ge(0))
        #     loss -= self.alpha * torch.sum(intercenter_dist) / dist_num

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
