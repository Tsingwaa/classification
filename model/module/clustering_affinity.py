import torch
import torch.nn as nn


class ClusteringAffinity(nn.Module):
    def __init__(self, num_classes, num_centers, sigma, feat_dim,
                 init_weight=True):
        """Affinity layer to replace fc layer

        refer to paper:
            Gaussian Affinity for Max-margin Class Imbalanced Learning.
            ICCV2019. Munawar Hayat et al.

        Args:
            num_classes(int, C): number of classes
            num_centers(int, m): number of centers
            sigma(list): variation per class
            feat_dim(int, d): dimensions of feature vector
            init_weight(bool): whether to init center with kaiming_normal
        """
        super(ClusteringAffinity, self).__init__()
        self.num_classes = num_classes
        self.num_centers = num_centers
        self.feat_dim = feat_dim
        self.sigma = sigma
        # shape: (C, m, d)
        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.num_centers, self.feat_dim),
            requires_grad=True
        )
        # self.my_registered_parameter=nn.ParameterList([self.centers])
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, f):
        # f: (N, d)

        N = f.size(0)  # batch size

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
        mc = self.num_centers * self.num_classes
        w_reshape = self.centers.view(mc, self.feat_dim)  # (m*C, d)
        w_reshape_expand1 = w_reshape.unsqueeze(0)  # (1, m*C, d)
        w_reshape_expand2 = w_reshape.unsqueeze(1)  # (m*C, 1, d)

        # (m*C, m*C, d) --> (m*C, m*C)
        w_norm = torch.sum((w_reshape_expand2 - w_reshape_expand1)**2, -1)
        triu_w_norm = torch.triu(w_norm, 1)  # strict upper triangle matrix

        mu = 2.0 / (mc**2 - mc) * triu_w_norm.sum()  # float

        residuals = torch.triu((triu_w_norm - mu)**2, 1)  # (m*C, m*C)
        rw = 2.0 / (mc**2 - mc) * residuals.sum()  # float

        # rw=residuals.sum()
        rw_broadcast = torch.ones((N, 1)).to('cuda') * rw  # (N, 1)

        output = torch.cat((distance, rw_broadcast), dim=-1)  # (N, C+1)

        return output
