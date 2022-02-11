from model.loss.builder import Losses
from torch import nn


@Losses.register_module("SimSiamLoss")
class SimSiamLoss(nn.Module):

    def __init__(self, **kwargs):
        super(SimSiamLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, p1, p2, z1, z2):
        self.cos_sim = self.cos_sim.cuda()

        loss1 = -0.5 * self.cos_sim(p1, z2).mean()
        loss2 = -0.5 * self.cos_sim(p2, z1).mean()

        return 1 + loss1 + loss2
