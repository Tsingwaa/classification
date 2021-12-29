import torch
import torch.nn as nn
# from torch.autograd.function import Function
# import torch.nn.functional as F
# from torch.autograd import Variable


class ClusteringAffinity(nn.Module):
    def __init__(self, n_classes, n_centers, sigma, feat_dim,
                 init_weight=True):
        """Affinity layer to replace fc layer

        refer to paper:
            Gaussian Affinity for Max-margin Class Imbalanced Learning.
            ICCV2019. Munawar Hayat et al.

        Args:
            n_classes(int, C): number of classes
            n_centers(int, m): number of centers
            sigma(list): variation per class
            feat_dim(int, d): dimensions of feature vector
            init_weight(bool): whether to init center with kaiming_normal
        """
        super(ClusteringAffinity, self).__init__()
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.feat_dim = feat_dim
        self.sigma = sigma
        # shape: (C, m, d)
        self.centers = nn.Parameter(
            torch.randn(self.n_classes, self.n_centers, self.feat_dim),
            requires_grad=True
        )
        # self.my_registered_parameter=nn.ParameterList([self.centers])
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, f):
        # f: (N, d)
        batch_size = f.size(0)

        # Euclidean space similarity measure, calculate d(f_i, w_j)
        f_expand = f.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, d)
        w_expand = self.centers.unsqueeze(0)  # (1, C, m, d)
        # 每个样本点的fv和所有类中心求距离
        # f_expand - w_expand = (N, C, m, d) --> (N, C, m)
        fw_norm = torch.sum((f_expand - w_expand) ** 2, -1)  # sum the last dim
        distance = torch.exp(- fw_norm / self.sigma)
        distance = torch.max(distance, -1)[0]  # (N, C, m) -> (N, C)
        # self.centers.zeros_grad()

        # Regularization
        mc = self.n_centers * self.n_classes
        w_reshape = self.centers.view(mc, self.feat_dim)  # (m*C, d)
        w_reshape_expand1 = w_reshape.unsqueeze(0)  # (1, m*C, d)
        w_reshape_expand2 = w_reshape.unsqueeze(1)  # (m*C, 1, d)
        # (m*C, m*C, d) --> (m*C, m*C)
        w_norm_mat = torch.sum((w_reshape_expand2 - w_reshape_expand1)**2, -1)
        w_norm_upper = torch.triu(w_norm_mat, 1)
        mu = 2.0/(mc**2 - mc) * w_norm_upper.sum()
        residuals = torch.triu((w_norm_upper - mu)**2, 1)
        rw = 2.0/(mc**2 - mc) * residuals.sum()
        # rw=residuals.sum()
        rw_broadcast = torch.ones((batch_size, 1)).to('cuda') * rw

        output = torch.cat((distance, rw_broadcast), dim=-1)

        return output

    # def upper_triangle(self, matrix):
    #     """严格上三角矩阵，取torch.triu(matrix, 1)"""
    #     upper = torch.triu(matrix)
    #     diagonal = torch.diag(matrix)
    #     diagonal_mask = torch.sign(torch.abs(torch.diag(diagonal)))
    #     return upper * (1.0 - diagonal_mask)
