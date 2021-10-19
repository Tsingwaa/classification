import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from data_loader.dataset.builder import Datasets


@Datasets.register_module("Caltech256-Trainset")
class Train_ImageFolder(ImageFolder):
    # Caltech256-5
    # mean = [0.4997, 0.5033, 0.4790]
    # std = [0.2794, 0.2621, 0.2800]

    # Calutech256-5-1
    mean = [0.5610, 0.5714, 0.5449]
    std = [0.2936, 0.4941, 0.3195]

    def __init__(self, data_root=None, transform=None, **kwargs):
        super(Train_ImageFolder, self).__init__(
            root=data_root,
            transform=transform,
        )
        self.cls_num = len(self.classes)

    def __getitem__(self, index):
        img_fpath, img_label = self.imgs[index]
        img = self.loader(img_fpath)
        img = self._check_channel(img)

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, img_label

    def _check_channel(self, Image_obj):
        img_arr = np.array(Image_obj)
        if len(img_arr.shape) < 3:
            img_arr_expand = np.repeat(img_arr[:, :, np.newaxis], 3, axis=2)
            Image_obj = Image.fromarray(img_arr_expand)
        return Image_obj


@Datasets.register_module("Caltech256-Valset")
class Val_ImageFolder(ImageFolder):
    # Caltech256-5
    # mean = [0.4997, 0.5033, 0.4790]
    # std = [0.2794, 0.2621, 0.2800]

    # Calutech256-5-1
    mean = [0.5610, 0.5714, 0.5449]
    std = [0.2936, 0.4941, 0.3195]

    def __init__(self, data_root=None, transform=None, **kwarg):
        super(Val_ImageFolder, self).__init__(
            root=data_root,
            transform=transform,
        )
        self.cls_num = len(self.classes)

    def __getitem__(self, index):
        img_fpath, img_label = self.imgs[index]
        img = self.loader(img_fpath)
        img = self._check_channel(img)

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, img_label, img_fpath

    def _check_channel(self, Image_obj):
        img_arr = np.array(Image_obj)
        if len(img_arr.shape) < 3:
            img_arr_expand = np.repeat(img_arr[:, :, np.newaxis], 3, axis=2)
            Image_obj = Image.fromarray(img_arr_expand)
        return Image_obj


@Datasets.register_module("Caltech256-Testset")
class Test_ImageFolder(ImageFolder):
    # Caltech256-5
    # mean = [0.4997, 0.5033, 0.4790]
    # std = [0.2794, 0.2621, 0.2800]

    # Caltech256-5-1
    mean = [0.5610, 0.5714, 0.5449]
    std = [0.2936, 0.4941, 0.3195]

    def __init__(self, data_root=None, transform=None, **kwarg):
        super(Test_ImageFolder, self).__init__(
            root=data_root,
            transform=transform,
        )
        self.cls_num = len(self.classes)

    def __getitem__(self, index):
        img_fpath, img_label = self.imgs[index]
        img = self.loader(img_fpath)
        img = self._check_channel(img)

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, img_label

    def _check_channel(self, Image_obj):
        img_arr = np.array(Image_obj)
        if len(img_arr.shape) < 3:
            img_arr_expand = np.repeat(img_arr[:, :, np.newaxis], 3, axis=2)
            Image_obj = Image.fromarray(img_arr_expand)
        return Image_obj
