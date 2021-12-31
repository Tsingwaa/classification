import torch
import torch.nn as nn
import torch.nn.functional as F
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
        loss = (-targets * log_probs).mean(0).sum()
        return loss


# @Losses.register_module("CrossEntropyLoss")
# class CELoss(nn.CrossEntropyLoss):
#     """This criterion computes the cross entropy loss between input and target.

#     It is useful when training a classification problem with `C` classes.
#     If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
#     assigning weight to each of the classes.
#     This is particularly useful when you have an imbalanced training set.
#     Args:
#         weight (Tensor, optional): a manual rescaling weight given to each
#             class. If given, has to be a Tensor of size `C`
#         reduction (string, optional): Specifies the reduction to apply to
#             the output: ``'none'`` | ``'mean'`` | ``'sum'``.
#             ``'none'``: no reduction will be applied
#             ``'mean'``: the weighted mean of the output is taken,
#             ``'sum'``: the output will be summed.
#             Default: ``'mean'``
#         label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the
#             amount of smoothing when computing the loss, where 0.0 means no
#             smoothing. The targets become a mixture of the original ground
#             truth and a uniform distribution as described in `Rethinking the
#             Inception Architecture for Computer Vision
#             <https://arxiv.org/abs/1512.00567>`.
#             Default: :math:`0.0`.
#     """

#     def __init__(self,
#                  weight=None,
#                  size_average=None,
#                  ignore_index=-100,
#                  reduce=None,
#                  reduction='mean',
#                  label_smoothing=0.0,
#                  **kwargs):
#         super(CELoss, self).__init__(weight=weight,
#                                      size_average=size_average,
#                                      ignore_index=ignore_index,
#                                      reduce=reduce,
#                                      reduction=reduction,
#                                      label_smoothing=label_smoothing)
        # self.weight = weight
        # self.reduction = reduction
        # self.label_smoothing = label_smoothing

    # def forward(self, input_, target):
    # if self.weight is not None:
    #     self.weight = self.weight.cuda()
    # return F.cross_entropy(input_, target,
    #                        weight=self.weight,
    #                        reduction=self.reduction,
    #                        label_smoothing=self.label_smoothing)

@Losses.register_module("CrossEntropyLoss")
class CELoss(nn.CrossEntropyLoss):
    """This criterion computes the cross entropy loss between input and target.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an imbalanced training set.
    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        reduction (string, optional): Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied
            ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed.
            Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the
            amount of smoothing when computing the loss, where 0.0 means no
            smoothing. The targets become a mixture of the original ground
            truth and a uniform distribution as described in `Rethinking the
            Inception Architecture for Computer Vision
            <https://arxiv.org/abs/1512.00567>`.
            Default: :math:`0.0`.
    """

    def __init__(self,
                 weight=None,
                 size_average=None,
                 ignore_index=-100,
                 reduce=None,
                 reduction='mean',
                 **kwargs):
        super(CELoss, self).__init__(weight=weight,
                                     size_average=size_average,
                                     ignore_index=ignore_index,
                                     reduce=reduce,
                                     reduction=reduction)

@Losses.register_module("BCEwithLogitsLoss")
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, X, y):
        loss = self.loss(X, y)
        return loss
