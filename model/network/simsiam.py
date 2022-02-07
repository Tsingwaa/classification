import torch
import torch.nn as nn


class ProjectionMLP(nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
        )
        self.l2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
        )
        self.l3 = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        x = self.l1(x)
        # x = self.l2(x)
        x = self.l3(x)

        return x


class PredictionMLP(nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
        )
        self.l2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x


class SimSiamModel(nn.Module):

    def __init__(self, fearureExactor):
        super(SimSiamModel, self).__init__()
        self.featureExactor = fearureExactor

        self.projection = ProjectionMLP(self.featureExactor.output_size, 2048,
                                        2048)
        self.prediction = PredictionMLP(2048, 512, 2048)

    def forward(self, x1, x2):
        x1 = self.featureExactor(x1)
        z1 = self.projection(x1)
        p1 = self.prediction(z1)

        with torch.no_grad():
            x2 = self.featureExactor(x2)
            z2 = self.projection(x2)

        return p1, z2
