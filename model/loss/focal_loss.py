import torch
import torch.nn as nn
import torch.nn.functional as F
from model.loss.builder import Losses


@Losses.register_module("FocalLoss")
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        CE_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * CE_loss).mean()

        return focal_loss
