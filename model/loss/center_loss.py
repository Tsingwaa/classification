"""
Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face
    Recognition. ECCV 2016.
"""
from __future__ import absolute_import

import torch
from torch import nn
from model.loss.builder import Losses
from utils import cos_sim, eu_dist


@Losses.register_module("CenterLoss")
class CenterLoss(nn.Module):

    def __init__(self, num_class=10, num_feature=2, alpha=0, **kwargs):
        """Initialize class centers
        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
        """

        super(CenterLoss, self).__init__()
        self.num_class = num_class
        self.num_feature = num_feature
        self.alpha = alpha
        self.centers = nn.Parameter(
            torch.randn(self.num_class, self.num_feature))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
            constraint: give the centers constraint to push each other away.
        """
        center = self.centers[labels]
        dist = (x - center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        if self.alpha > 0:
            intercenter_dist = eu_dist(self.centers, self.centers)
            intercenter_dist = torch.triu(intercenter_dist, diagonal=1)
            # print(torch.sum(intercenter_dist))
            dist_num = torch.sum(intercenter_dist.ge(0))
            loss -= self.alpha * torch.sum(intercenter_dist) / dist_num

        return loss

# class CenterLoss(nn.Module):
#     """Reference: Wen et al. A Discriminative Feature Learning Approach for
#     Deep Face Recognition. ECCV 2016.

#     Args:
#         num_classes (int): number of classes.
#         feat_dim (int): feature dimension.
#     """

#     def __init__(self, num_classes=10, feat_dim=2, use_gpu=True, **kwargs):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_gpu = use_gpu

#         self.centers = nn.Parameter(
#             torch.randn(self.num_classes, self.feat_dim))

#         if self.use_gpu:
#             self.centers = self.centers.cuda()

#     def forward(self, x, labels):
#         """
#         Args:
#             x: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (num_classes).
#         """
#         assert x.size(0) == labels.size(0), "features size != labels size"

#         batch_size = x.size(0)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True)\
#             .expand(batch_size, self.num_classes) +\
#             torch.pow(self.centers, 2).sum(dim=1, keepdim=True)\
#             .expand(self.num_classes, batch_size).t()

#         distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

#         classes = torch.arange(self.num_classes).long()
#         if self.use_gpu:
#             classes = classes.cuda()

#         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask = labels.eq(classes.expand(batch_size, self.num_classes))

#         dist = distmat * mask.float()
#         loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

#         return loss


if __name__ == '__main__':
    use_gpu = False
    num_classes = 10
    feat_dim = 2

    features = torch.ones(16, 2)
    targets = torch.ones((16,)).long()
    if use_gpu:
        features = features.cuda()
        targets = targets.cuda()

    center_loss = CenterLoss(num_classes, feat_dim, use_gpu)

    loss = center_loss(features, targets)
    print(loss)
