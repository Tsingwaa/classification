import torch
from torch import nn
from torchvision import models


class BaseNetwork(nn.Module):

    def __init__(self, arch, num_classes, pretrained=False, **kwargs):
        super(BaseNetwork, self).__init__()

        self.num_classes = num_classes

        if arch == "ResNet50":
            backbone = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.fc_in = 2048
        elif arch == "MobileNetv2":
            backbone = models.mobilenet_v2(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.fc_in = 1280
        elif arch == "DenseNet121":
            backbone = models.densenet121(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.fc_in = 1024

        self.fc = nn.Linear(self.fc_in, num_classes)

    def extract(self, x):
        feat_map = self.features(x)
        feat_vec = torch.flatten(feat_map, 1)
        return feat_vec

    def forward(self, x, out_type="fc"):
        """Return according to the out_type(default: "fc"-->logits)"""
        feat_vec = self.extract(x)
        if "vec" in out_type:
            return feat_vec
        else:
            logits = self.fc(feat_vec)
            return logits
