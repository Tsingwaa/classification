import torch
from torch import nn
# from pudb import set_trace
from model.network.builder import Networks


@Networks.register_module('MLP')
class MLP(nn.Module):
    def __init__(self, in_channels, num_classes, layers,
                 activate_layer='relu'):
        if activate_layer == 'sigmoid':
            self.activate = nn.Sigmoid(inplace=True)
        else:
            self.activate = nn.ReLU(inplace=True)

        self.mlp = nn.Sequential()

        for i in range(1, len(layers)):
            self.mlp.add_module(f'linear{i}', nn.Linear(layers[i-1], layers[i]))
            self.mlp.add_module(f'relu{i}', self.activate)
    

