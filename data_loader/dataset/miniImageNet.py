"""Adopted from https://github.com/Megvii-Nanjing/BBN
Customized by Kaihua Tang
"""

# import os
import math
import torch
import numpy as np
import imghdr
# from pudb import set_trace
from os.path import join
from PIL import Image
# from torchvision import transforms
from torchvision.datasets import ImageFolder
from data_loader.dataset.builder import Datasets
# TiffImagePlugin.DEBUG = True


@Datasets.register_module('imb_miniImageNet')
class ImbalanceMiniImageNet(torch.utils.data.Dataset):
    """Custom miniImageNet
    num_classes: 100
    num_samples_per_cls:
        - train: 1000
        - val: 200
        - test: 100
    size: larger than 224x224
    """

    # ImageNet
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Full miniImageNet
    mean = [0.4153, 0.4586, 0.4759]
    std = [0.2955, 0.2785, 0.2847]

    num_classes = 100

    # set_trace()
    def __init__(self, data_root, phase, img_lst_fpath=None, transform=None,
                 imb_type='exp', imb_factor=0.1, seed=0, adapt=False,
                 **kwargs):
        self.img_paths = []
        self.targets = []
        self.transform = transform
        self.adapt = adapt
        self.phase = phase

        # build img_path-target pairs
        if img_lst_fpath is not None and '/' not in img_lst_fpath:
            img_lst_fpath = join(data_root, img_lst_fpath)
        else:
            img_lst_fpath = join(data_root, phase + '.txt')
        with open(img_lst_fpath) as f:
            for line in f:
                img_path = join(data_root, line.split()[0][1:])
                # line starts with '/', so index by [1:]
                if imghdr.what(img_path) == '.tiff':
                    print(img_path)
                self.img_paths.append(img_path)
                self.targets.append(int(line.split()[1]))

        self.img_paths = np.array(self.img_paths)
        self.num_samples_per_cls = []

        if phase == 'train':
            np.random.seed(seed)
            # generate imbalance num_samples list
            self.num_samples_per_cls = self.get_num_samples_per_cls(
                self.num_classes, imb_type, imb_factor
            )
            # regenerate self.img_paths and self.targets
            self.gen_imbalanced_data(self.num_samples_per_cls)
            self.class_weight = self.get_class_weight()

        self.classes = [self.label2ctg[i] for i in range(self.num_classes)]

    def get_num_samples_per_cls(self, num_classes, imb_type, imb_factor):
        """Generate imbalanced num samples by 'exp' or 'step'.
        - imb_type: 'exp' or 'step'
        - imb_factor: (default: 0.1) the largest / the smallest
        - steps: if imb_type is 'step', how many steps.
        """
        max_num_samples = len(self.img_paths) / num_classes
        num_samples_per_cls = []
        if imb_type == 'exp':  # exponential moving
            for cls_idx in range(num_classes):
                img_num = max_num_samples * imb_factor ** (
                    cls_idx / (num_classes - 1))
                num_samples_per_cls.append(int(img_num))
        elif imb_type == 'step':  # two different num_samples
            head_classes = math.floor(num_classes / 3)  # head=tail
            tail_classes = head_classes
            # 3step, 20classes： 7-7-6
            for cls_idx in range(num_classes):
                if cls_idx < head_classes:
                    step = 0
                elif head_classes <= cls_idx < num_classes - tail_classes:
                    step = 1
                else:
                    step = 2
                img_num = max_num_samples * imb_factor ** (step / 2)
                num_samples_per_cls.append(int(img_num))
        else:
            num_samples_per_cls.extend([int(max_num_samples)] * num_classes)

        return num_samples_per_cls

    def gen_imbalanced_data(self, num_samples_per_cls):
        new_img_paths = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.cls2nsamples = dict()
        for cls, n_samples in zip(classes, num_samples_per_cls):
            self.cls2nsamples[cls] = n_samples
            # random shuffle indexs and select num_samples of the class
            idx = np.where(targets_np == cls)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:n_samples]
            # generate new train pairs
            new_img_paths.extend(self.img_paths[selec_idx].tolist())
            new_targets.extend([cls, ] * n_samples)

        self.img_paths = new_img_paths
        self.targets = new_targets

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        target = self.targets[index]
        img = Image.open(img_path).convert('RGB')
        img = self._check_channel(img)  # 对单通道灰度图复制为三通道

        if self.transform is not None:
            # if self.adapt and (self.phase == 'train'):
            #     percent = np.sum(self.class_weight[:(target+1)])
            # else:
            #     percent = None
            img = self.transform(img, mean=self.mean, std=self.std,
                                 percent=None)

        return img, target

    def __len__(self):
        return len(self.targets)

    @property
    def label2ctg(self):
        label2ctg = dict()
        for i in range(len(self.targets)):
            if self.targets[i] not in label2ctg:
                category = self.img_paths[i].split('/')[-2]
                label2ctg[self.targets[i]] = category
        return label2ctg

    def get_class_weight(self):
        samplers_per_cls = np.array(self.num_samples_per_cls)
        weight = 1.0 / samplers_per_cls
        weight /= np.sum(weight)
        return weight

    def _check_channel(self, Image_obj):
        img_arr = np.array(Image_obj)
        if len(img_arr.shape) < 3:
            img_arr_expand = np.repeat(img_arr[:, :, np.newaxis], 3, axis=2)
            Image_obj = Image.fromarray(img_arr_expand)
        return Image_obj


@Datasets.register_module('imb_miniImageNet20')
class ImbalanceMiniImageNet20(ImbalanceMiniImageNet):
    """Select the former 20 classes as a small imbalanced dataset
    for test use
    """
    mean = [0.3567, 0.4279, 0.4439]
    std = [0.2761, 0.2584, 0.2705]
    num_classes = 20


@Datasets.register_module('imb_miniImageNet3')
class ImbalanceMiniImageNet3(ImbalanceMiniImageNet):
    """Select the former 20 classes as a small imbalanced dataset
    for test use
    """
    mean = [0.3644, 0.4094, 0.3998]
    std = [0.2698, 0.2434, 0.2535]
    num_classes = 3


@Datasets.register_module('imb_miniImageNet20_tail')
class ImbalanceMiniImageNet20_tail(ImbalanceMiniImageNet):
    """Select the former 20 classes as a small imbalanced dataset
    for test use
    """
    mean = [0.3567, 0.4279, 0.4439]
    std = [0.2761, 0.2584, 0.2705]
    num_classes = 20

    def __init__(self, data_root, phase, img_lst_fpath=None, transform=None,
                 imb_type='exp', imb_factor=0.1, seed=0, tail_num_classes=0,
                 **kwargs):
        self.img_paths = []
        self.targets = []
        self.transform = transform

        # build img_path-target pairs
        if img_lst_fpath is not None and '/' not in img_lst_fpath:
            img_lst_fpath = join(data_root, img_lst_fpath)
        else:
            img_lst_fpath = join(data_root, phase + '.txt')
        with open(img_lst_fpath) as f:
            for line in f:
                img_path = join(data_root, line.split()[0][1:])
                # line starts with '/', so index by [1:]
                if imghdr.what(img_path) == '.tiff':
                    print(img_path)
                self.img_paths.append(img_path)
                self.targets.append(int(line.split()[1]))

        self.img_paths = np.array(self.img_paths)
        self.img_num = []

        if phase == 'train':
            np.random.seed(seed)
            # generate imbalance num_samples list
            img_num_list = self.get_num_samples_per_cls(
                self.num_classes, imb_type, imb_factor
            )
            # regenerate self.img_paths and self.targets
            self.gen_imbalanced_data(img_num_list)
            self.img_num = img_num_list

        if tail_num_classes:
            self.targets = np.array(self.targets)
            self.img_paths = np.array(self.img_paths)
            tail_indexes = np.where(
                self.targets >= (self.num_classes - tail_num_classes))
            self.targets = self.targets[tail_indexes]
            self.img_paths = self.img_paths[tail_indexes]
            self.num_classes = tail_num_classes
        self.labels = self.targets
        label2ctg = self.get_label2ctg()
        self.classes = [label2ctg[tail_num_classes+i]
                        for i in range(self.num_classes)]
        self.img_num = self.img_num[-tail_num_classes:]


@Datasets.register_module("miniImageNet")
class miniImageNet(ImageFolder):
    # Full miniImageNet
    mean = [0.4153, 0.4586, 0.4759]
    std = [0.2955, 0.2785, 0.2847]
    num_classes = 100

    def __init__(self, data_root, phase, transform=None, **kwargs):
        super(miniImageNet, self).__init__(
            root=join(data_root, phase),
            transform=transform,
        )
        self.num_classes = len(self.classes)

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


@Datasets.register_module("miniImageNet_eval")
class miniImageNet_eval(ImageFolder):
    # Full miniImageNet
    mean = [0.4153, 0.4586, 0.4759]
    std = [0.2955, 0.2785, 0.2847]
    num_classes = 100

    def __init__(self, data_root, phase, transform=None, **kwarg):
        super(miniImageNet_eval, self).__init__(
            root=join(data_root, phase),
            transform=transform,
        )
        self.num_classes = len(self.classes)

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
