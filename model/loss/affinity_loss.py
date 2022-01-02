import torch
import torch.nn as nn
from .builder import Losses


@Losses.register_module('Affinity_Loss')
class Affinity_Loss(nn.Module):
    def __init__(self, lamda, **kwargs):
        super(Affinity_Loss, self).__init__()
        self.lamda = lamda  # control margin

    def forward(self, y_pred_plusone, y_true_plusone):
        """loss forward

        Args:
            y_pred_plusone(tensor, (N, C+1)): distance + rw_broadcast
            y_true_plusone(tensor, (N, C+1)): add dummy 1-d zeros
        """
        batch_size = y_true_plusone.size(0)

        onehot = y_true_plusone[:, :-1]  # onehot + 0
        distance = y_pred_plusone[:, :-1]  # distance (N, C)
        rw = torch.mean(y_pred_plusone[:, -1])  # (N, 1) --> (1)

        # (N, C)->(N,)->(N, 1)
        d_fi_wyi = torch.sum(onehot * distance, -1).unsqueeze(1)

        # distance-d_f_w = (N, C) - (N, 1) --> (N, C)
        losses = torch.clamp(self.lamda + distance - d_fi_wyi, min=0)

        # (N, 1) / N
        L_mm = torch.sum((1.0 - onehot) * losses, -1) / batch_size

        avg_loss = torch.sum(L_mm + rw, -1)

        return avg_loss
