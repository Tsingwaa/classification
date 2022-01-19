import os

import base
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from torchvision import datasets, transforms


@Datasets.register_module("MNIST")
class MnistDataLoader(base.BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    num_classes = 10

    def __init__(self,
                 root,
                 batch_size,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 training=True):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        root = os.path.join(DATASETS_ROOT, root)
        self.dataset = datasets.MNIST(root,
                                      train=training,
                                      download=True,
                                      transform=transformer)
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers)
