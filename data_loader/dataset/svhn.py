import torchvision
from data_loader.dataset.builder import Datasets


@Datasets.register_module("SVHN")
class SVHN_(torchvision.datasets.SVHN):
    def __init__(self, data_root, train=True, transform=None, download=True):
        super(SVHN_, self).__init__(
            root=data_root,
            split='train' if train else 'test',
            transform=transform,
            download=download
        )
