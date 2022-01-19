import os

import torchvision
from data_loader.dataset.builder import DATASETS_ROOT, Datasets


@Datasets.register_module("SVHN")
class SVHN_(torchvision.datasets.SVHN):
    num_classes = 10

    def __init__(self, root, train, transform=None, download=True):
        root = os.path.join(DATASETS_ROOT, root)
        super(SVHN_, self).__init__(root=root,
                                    split='train' if train else 'test',
                                    transform=transform,
                                    download=download)
