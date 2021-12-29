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
            n_classes(int): number of classes
            n_centers(int): number of centers
            sigma(list): variation per class
            feat_dim(int): dimensions of feature vector
            init_weight(bool): whether to init center with kaiming_normal
        """
        super(ClusteringAffinity, self).__init__()
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.feat_dim = feat_dim
        self.sigma = sigma
        self.centers = nn.Parameter(
            torch.randn(self.n_classes, self.n_centers, self.feat_dim),
            requires_grad=True
        )
        # self.my_registered_parameter=nn.ParameterList([self.centers])
        if init_weight:
            self.__init_weight()

    # init the centers
    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, f):
        f_expand = f.unsqueeze(1).unsqueeze(1)
        w_expand = self.centers.unsqueeze(0)
        fw_norm = torch.sum((f_expand-w_expand)**2, -1)
        distance = torch.exp(-fw_norm/self.sigma)
        distance = torch.max(distance, -1)[0]
        # self.centers.zeros_grad()
        # Regularization
        mc = self.n_centers*self.n_classes
        w_reshape = self.centers.view(mc, self.feat_dim)
        w_reshape_expand1 = w_reshape.unsqueeze(0)
        w_reshape_expand2 = w_reshape.unsqueeze(1)
        w_norm_mat = torch.sum((w_reshape_expand2-w_reshape_expand1)**2, -1)
        w_norm_upper = self.upper_triangle(w_norm_mat)
        mu = 2.0/(mc**2-mc)*w_norm_upper.sum()
        residuals = self.upper_triangle((w_norm_upper-mu)**2)
        rw = 2.0/(mc**2-mc)*residuals.sum()
        # rw=residuals.sum()
        batch_size = f.size(0)
        rw_broadcast = torch.ones((batch_size, 1)).to('cuda')*rw
        output = torch.cat((distance, rw_broadcast), dim=-1)
        return output

    def upper_triangle(self, metrix):
        upper = torch.triu(metrix)
        diagonal = torch.diag(metrix)
        diagonal_mask = torch.sign(torch.abs(torch.diag(diagonal)))
        return upper*(1.0-diagonal_mask)


class Affinity_Loss(nn.Module):
    def __init__(self, lambd):
        super(Affinity_Loss, self).__init__()
        self.lamda = lambd

    def forward(self, y_pred_plusone, y_true_plusone):
        onehot = y_true_plusone[:, :-1]
        distance = y_pred_plusone[:, :-1]
        rw = torch.mean(y_pred_plusone[:, -1])
        d_fi_wyi = torch.sum(onehot*distance, -1).unsqueeze(1)
        losses = torch.clamp(self.lamda+distance-d_fi_wyi, min=0)
        L_mm = torch.sum(losses*(1.0-onehot), -1) / y_true_plusone.size(0)
        loss = torch.sum(L_mm+rw, -1)
        return loss
