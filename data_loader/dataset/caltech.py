from torchvision.datasets import ImageFolder
from data_loader.dataset.builder import Datasets


@Datasets.register_module("Caltech256-5-Trainset")
class Train_ImageFolder(ImageFolder):
    mean = [0.4997, 0.5033, 0.4790]
    std = [0.2794, 0.2621, 0.2800]

    def __init__(self, data_root=None, transform=None, **kwargs):
        super(ImageFolder).__init__(
            root=data_root,
            transform=transform,
        )

    def __getitem__(self, index):
        img_fpath, img_label = self.imgs[index]
        img = self.loader(img_fpath)

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, img_label


@Datasets.register_module("Caltech256-5-Valset")
class Val_ImageFolder(ImageFolder):
    mean = [0.4997, 0.5033, 0.4790]
    std = [0.2794, 0.2621, 0.2800]

    def __init__(self, data_root=None, transform=None, **kwarg):
        super(ImageFolder).__init__(
            root=data_root,
            transform=transform,
        )

    def __getitem__(self, index):
        img_fpath, img_label = self.imgs[index]
        img = self.loader(img_fpath)

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, img_label, img_fpath


@Datasets.register_module("Caltech256-5-Testset")
class TestDataset(ImageFolder):
    mean = [0.4997, 0.5033, 0.4790]
    std = [0.2794, 0.2621, 0.2800]

    def __init__(self, data_root=None, transform=None, **kwarg):
        super(ImageFolder).__init__(
            root=data_root,
            transform=transform,
        )

    def __getitem__(self, index):
        img_fpath, img_label = self.imgs[index]
        img = self.loader(img_fpath)

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, img_label
