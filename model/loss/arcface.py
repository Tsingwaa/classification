import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from common.loss.builder import Losses


@Losses.register_module("ArcCos")
class ArcCos(nn.Module):
    def __init__(
            self,
            in_features=128,
            out_features=100,
            s=30.0,
            m=0.50,
            bias=False,
            is_half=False):
        super(ArcCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.is_half = is_half
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if self.is_half:
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1)).half()
        else:
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ----------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')  # noqa
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


@Losses.register_module("SubCenterArcFace")
class SubCenterArcFace(nn.Module):
    """
    Implementation of
    `Sub-center ArcFace: Boosting Face Recognition
    by Large-scale Noisy Web Faces`_.
    .. _Sub-center ArcFace\\: Boosting Face Recognition \
        by Large-scale Noisy Web Faces:
        https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature,
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
        k: number of possible class centroids.
            Default: ``3``.
        eps (float, optional): operation accuracy.
            Default: ``1e-6``.
    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\\_features`.
    Example:
        >>> layer = SubCenterArcFace(5, 10, s=1.31, m=0.35, k=2)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()
    """

    def __init__(
            self,
            in_features=128,
            out_features=100,
            s=32.0,
            m=0.5,
            k=3,
            eps=1e-6,
            is_half=False):
        super(SubCenterArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        self.k = k
        self.eps = eps
        self.is_half = is_half
        self.weight = Parameter(torch.Tensor(k, in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.threshold = math.pi - self.m

    def __repr__(self):
        """Object representation."""
        rep = (
            "SubCenterArcFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m},"
            f"k={self.k},"
            f"eps={self.eps}"
            ")"
        )
        return rep

    def forward(self, input, target=None):
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.
        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes.
        """
        feats = (
            F.normalize(input).unsqueeze(0).expand(
                self.k, *input.shape))  # k*b*f
        wght = F.normalize(self.weight, dim=1)  # k*f*c
        if self.is_half:
            wght = wght.half()
        cos_theta = torch.bmm(feats, wght)  # k*b*f
        cos_theta = torch.max(cos_theta, dim=0)[0]  # b*f
        theta = torch.acos(
            torch.clamp(
                cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        if target is None:
            return cos_theta

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        selected = torch.where(
            theta > self.threshold,
            torch.zeros_like(one_hot),
            one_hot)

        logits = torch.cos(
            torch.where(
                selected.type(
                    torch.bool),
                theta +
                self.m,
                theta))
        logits *= self.s

        return logits


if __name__ == '__main__':
    loss_fn = nn.CrossEntropyLoss().cuda()
    embedding = torch.randn(3, 5, requires_grad=True).cuda()
    target = torch.empty(3, dtype=torch.long).random_(10).cuda()

    print('full')
    layer = SubCenterArcFace(
        in_features=5,
        out_features=10,
        s=1.31,
        m=0.35,
        k=2).cuda()
    print(embedding.size(), type(embedding), target.size(), type(target))
    output = layer(embedding, target)
    print(output.size(), type(output))
    loss = loss_fn(output, target)
    # loss.backward()
    print(loss)

    print('half')
    layer = SubCenterArcFace(
        in_features=5,
        out_features=10,
        s=1.31,
        m=0.35,
        k=2,
        is_half=True).cuda()
    embedding = embedding.half()
    print(embedding.size(), type(embedding), target.size(), type(target))
    output = layer(embedding, target)
    print(output.size(), type(output))
    loss = loss_fn(output, target)
    # loss.backward()
    print(loss)
