from torch import nn
# from pudb import set_trace
from model.network.builder import Networks


def weight_init(model):
    for m in model.modules():
        classname = m.__class__.__name__
        if classname == 'Linear':
            nn.init.kaiming_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.0)


@Networks.register_module('MLP2')
class MLP2(nn.Module):
    def __init__(self, in_channels, num_classes, activate_layer='ReLU',
                 **kwargs):
        super(MLP2, self).__init__()
        assert activate_layer == 'ReLU'
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, int(in_channels/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels/2), num_classes),
        )
        # weight_init(self)  # Default kaiming_normal_

    def forward(self, x):
        return self.mlp(x)


@Networks.register_module('MLP3')
class MLP3(nn.Module):
    def __init__(self, in_channels, num_classes, activate_layer='ReLU',
                 **kwargs):
        super(MLP3, self).__init__()
        assert activate_layer == 'ReLU'
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, int(in_channels/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels/2), int(in_channels/4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels/4), num_classes),
        )
        # weight_init(self)

    def forward(self, x):
        return self.mlp(x)


@Networks.register_module('MLP4')
class MLP4(nn.Module):
    def __init__(self, in_channels, num_classes, activate_layer='ReLU',
                 **kwargs):
        super(MLP4, self).__init__()
        assert activate_layer == 'ReLU'
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, int(in_channels/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels/2), int(in_channels/4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels/4), int(in_channels/8)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels/8), num_classes),
        )
        # weight_init(self.mlp)

    def forward(self, x):
        return self.mlp(x)
