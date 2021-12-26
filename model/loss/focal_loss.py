import torch
import torch.nn as nn
import torch.nn.functional as F
from model.loss.builder import Losses


@Losses.register_module("FocalLoss")
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, **kwargs):
        super(FocalLoss, self).__init__()
        self.weight = weight  # 1*C
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # N*1
        pt = torch.exp(-ce_loss)  # N*1: softmax prob of the target class
        if self.weight is not None:
            self.weight = self.weight.cuda()
            ce_loss *= self.weight[targets]  # (N*1) * (N*1)
        focal_loss = ((1 - pt)**self.gamma * ce_loss).mean()  # 1*1
        return focal_loss
