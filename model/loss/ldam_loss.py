import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pudb import set_trace
from .builder import Losses


@Losses.register_module('LDAM_Loss')
class LDAMLoss(nn.Module):
    def __init__(self, num_samples_per_cls, max_m=0.5, weight=None, s=30,
                 **kwargs):
        """ Init LDAM loss

        Args:
            num_samples_per_cls:
            max_m:
            weight:
            s:
        """
        super(LDAMLoss, self).__init__()

        # Compute \gamma_j = C / N_j^{1/4}

        # 1 / N_j^{1/4}
        m_list = 1.0 / np.sqrt(np.sqrt(num_samples_per_cls))
        # C = max_m / np.max(m_list), max_m is hyperparameter
        m_list = m_list * (max_m / np.max(m_list))

        self.m_list = torch.cuda.FloatTensor(m_list).unsqueeze(0)  # (1, C)

        assert s > 0
        self.s = s  # ?
        self.weight = weight  # standard reweight

    def forward(self, probs, target):
        """forward.

        Args:
            probs: (N, C)
            target: (N)
        """
        index = torch.zeros_like(probs, dtype=torch.uint8)  # zeros (N, C)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot (N, C)

        index_float = index.type(torch.cuda.FloatTensor)  # Cuda Float
        # tensor[None, :] is equal to tensor.unsqueeze(0)
        # (1, C) * (C, N) = (1, N)
        batch_m = torch.matmul(self.m_list, index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))  # (N, 1)
        x_m = probs - batch_m  # (N, C) - (N, 1) <== Broadcast

        # replace probs with x_m by index
        output = torch.where(index, x_m, probs)

        return F.cross_entropy(self.s * output, target, weight=self.weight)


# if __name__ == '__main__':
#     num_samples_per_cls = [1000] * 10
#     criterion = LDAMLoss(num_samples_per_cls).cuda()

#     probs = torch.randn((2, 10)).cuda()
#     targets = torch.randint(high=9, size=(2,)).cuda()

#     loss = criterion(probs, targets)

#     print(loss)
