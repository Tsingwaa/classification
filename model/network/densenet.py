import torch
import torch.nn as nn
import torch.nn.functional as F
from model.network.builder import Networks
from torchvision import models


@Networks.register_module("DenseNet121")
class DenseNet121(nn.Module):

    def __init__(self, num_classes, pretrained, **kwargs):
        super(DenseNet121, self).__init__()
        self.num_classes = num_classes

        backbone = models.densenet121(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc_in = 1024

        self.fc = nn.Linear(self.fc_in, num_classes)

        if kwargs.get("pred_head", False):
            self.pred_head = nn.Sequential(
                nn.Linear(self.fc_in, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
            )

    def extract(self, x):
        feat_map = self.features(x)
        feat_map = F.relu(feat_map, inplace=True)
        feat_map = F.adaptive_avg_pool2d(feat_map, (1, 1))
        feat_vec = torch.flatten(feat_map, 1)
        return feat_vec

    def forward(self, x1, x2=None, out_type="fc"):
        if out_type in ["supcon", "simclr"]:
            feat_vec = self.extract(x1)
            logits = self.fc(feat_vec)
            norm_vec = F.normalize(self.pred_head(feat_vec), dim=1)
            return logits, norm_vec

        elif out_type == "pred_head":
            x = self.extract(x1)
            x = self.pred_head(x)
            return F.normalize(x, dim=1)

        else:
            x1 = self.extract(x1)
            if 'fc' in out_type:
                return self.fc(x1)
            elif "vec" in out_type:
                return x1
            else:
                raise TypeError
