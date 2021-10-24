import torch
import torch.nn as nn
from model.loss.builder import Losses


@Losses.register_module("CrossEntropyLabelSmooth")
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference: Rethinking the Inception Architecture for Computer Vision.
    Szegedy et al. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=10, epsilon=0.1, use_gpu=True, **kwargs):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape
                     (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size())\
            .scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.cuda()
        targets = (1 - self.epsilon) * targets +\
            self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


@Losses.register_module("CrossEntropyLoss")
class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X, y):
        y = y.cuda()
        loss = self.loss(X, y)
        return loss


@Losses.register_module("BCEwithLogitsLoss")
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, X, y):
        y = y.cuda()
        loss = self.loss(X, y)
        return loss
