import os

import numpy as np
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from PIL import Image
from torchvision.datasets import ImageFolder

CALTECH256_MEAN_STD_DICT = {
    5: ([0.4790, 0.5033, 0.4997], [0.2800, 0.2621, 0.2794]),
    10: ([0.4863, 0.5066, 0.5205], [0.3215, 0.3114, 0.3200])
}


@Datasets.register_module("Caltech256-Trainset")
class Train_ImageFolder(ImageFolder):

    def __init__(self, root, transform=None, **kwargs):
        root = os.path.join(DATASETS_ROOT, root)
        super(Train_ImageFolder, self).__init__(
            root=root,
            transform=transform,
        )
        self.cls_num = len(self.classes)
        self.labels = [x[1] for x in self.imgs]
        self.mean, self.std = CALTECH256_MEAN_STD_DICT[self.cls_num]

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

    def __init__(self, root=None, transform=None, **kwarg):
        root = os.path.join(DATASETS_ROOT, root)
        super(Val_ImageFolder, self).__init__(
            root=root,
            transform=transform,
        )
        self.cls_num = len(self.classes)
        self.labels = [x[1] for x in self.imgs]
        self.mean, self.std = CALTECH256_MEAN_STD_DICT[self.cls_num]

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

    def __init__(self, data_root=None, transform=None, **kwarg):
        super(Test_ImageFolder, self).__init__(
            root=data_root,
            transform=transform,
        )
        self.cls_num = len(self.classes)
        self.labels = [x[1] for x in self.imgs]
        self.mean, self.std = CALTECH256_MEAN_STD_DICT[self.cls_num]

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
