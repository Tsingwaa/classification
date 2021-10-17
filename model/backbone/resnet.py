from torchvision.models import ResNet
from model.backbone.builder import Backbones


@Backbones.register_module('ResNet34')
class ResNet34(ResNet):
    def __init__(self, num_classes, layers=[3, 4, 6, 3], pretrained=False):

