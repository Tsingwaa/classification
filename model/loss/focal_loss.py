import torch
import torch.nn as nn
import torch.nn.functional as F
from model.loss.builder import Losses


@Losses.register_module("FocalLoss")
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight
        # weight parameter will act as the alpha parameter
        # to balance class weights

    def forward(self, input, target):

        CE_loss = F.cross_entropy(input, target,
                                  reduction=self.reduction,
                                  weight=self.weight)
        pt = torch.exp(-CE_loss)
        focal_loss = ((1 - pt) ** self.gamma * CE_loss).mean()

        return focal_loss
