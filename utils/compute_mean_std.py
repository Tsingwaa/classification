###############################################################################
# Copyright (C) 2021 All rights reserved.
# Filename: compute_mean_std.py
# Author: Tsingwaa
# Email: zengchh3@gmail.com
# Created Time : 2021-12-20 19:27 Monday
# Last modified: 2021-12-20 19:27 Monday
# Description:
#
###############################################################################

import math
from os.path import join

import numpy as np
import torch
from PIL import Image
# from pudb import set_trace
# import torchvision
from tqdm import tqdm


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
    # mean = [0.4759, 0.4586, 0.4153]
    # std = [0.2847, 0.2785, 0.2955]

    cls_num = 3

    def __init__(self,
                 data_root,
                 phase,
                 img_lst_fpath,
                 transform=None,
                 imb_type='exp',
                 imb_factor=0.01,
                 seed=0,
                 **kwargs):
        self.img_paths = []
        self.targets = []
        self.transform = transform
        # build img_path-target pairs
        img_lst_fpath = join(data_root, img_lst_fpath)
        with open(img_lst_fpath) as f:
            for line in f:
                img_path = join(data_root, line.split()[0][1:])
                # line starts with '/', so index by [1:]
                self.img_paths.append(img_path)
                self.targets.append(int(line.split()[1]))

        self.img_paths = np.array(self.img_paths)
        self.img_num = []

        if phase == 'train':
            np.random.seed(seed)
            # generate imbalance num_samples list
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type,
                                                    imb_factor)
            # according to the generated img_samples,
            # generate new self.img_paths and self.targets
            self.gen_imbalanced_data(img_num_list)
            self.img_num = img_num_list

        self.labels = self.targets

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        """Generate imbalanced num samples by 'exp' or 'step'.
        - imb_type: 'exp' or 'step'
        - imb_factor: (default: 0.1) the largest / the smallest
        """
        img_max = len(self.img_paths) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':  # exponential moving
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':  # two different num_samples
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_img_paths = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            # random shuffle indexs and select num_samples of the class
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            # generate new train pairs
            new_img_paths.extend(self.img_paths[selec_idx].tolist())
            new_targets.extend([
                the_class,
            ] * the_img_num)

        self.img_paths = new_img_paths
        self.targets = new_targets

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        # img = Image.open(img_path).convert('RGB')

        return img_path, label

    def __len__(self):
        return len(self.labels)


def compute_mean_and_std(img_paths):
    # 输入PyTorch的数据路径列表，输出均值和标准差

    mean_r = 0.
    mean_g = 0.
    mean_b = 0.
    print("===> Computing mean...")
    for img_path in tqdm(img_paths, ncols=80):
        img = Image.open(img_path)  # 默认打开的通道为BGR
        img = np.asarray(img)  # change PIL Image to numpy array
        if len(img.shape) < 3 or img.shape[2] < 3:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(img_paths)
    mean_g /= len(img_paths)
    mean_r /= len(img_paths)

    mean = (mean_r.item() / 255.0, mean_g.item() / 255.0,
            mean_b.item() / 255.0)

    print('===> mean: [{:.4f},{:.4f},{:.4f}]\n'.format(mean[0], mean[1],
                                                       mean[2]))

    diff_r = 0.
    diff_g = 0.
    diff_b = 0.

    N = 0.
    print("===> Computing std...")
    for img_path in tqdm(img_paths, ncols=80):
        img = Image.open(img_path)  # 默认打开的通道为BGR
        img = np.asarray(img)
        if len(img.shape) < 3 or img.shape[2] < 3:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_r = np.sqrt(diff_r / N)
    std_g = np.sqrt(diff_g / N)
    std_b = np.sqrt(diff_b / N)

    std = (std_r.item() / 255.0, std_g.item() / 255.0, std_b.item() / 255.0)

    print('===> std: [{:.4f},{:.4f},{:.4f}]'.format(std[0], std[1], std[2]))

    return mean, std


def get_LT_img_paths(img_paths, targets, imb_type, imb_factor):
    """Generate imbalanced num samples by 'exp' or 'step'.
    - imb_type: 'exp' or 'step'
    - imb_factor: (default: 0.1) the largest / the smallest
    - steps: if imb_type is 'step', how many steps.
    """
    classes = np.unique(targets)
    num_classes = len(classes)
    img_max = len(img_paths) / num_classes
    num_per_cls = []  # 每个类别的样本数量
    if imb_type == 'exp':  # exponential moving
        for cls_idx in range(num_classes):
            img_num = img_max * imb_factor**(cls_idx / (num_classes - 1))
            num_per_cls.append(int(img_num))
    elif imb_type == 'step':  # two different num_samples
        head_classes = math.floor(num_classes / 3)  # head=tail
        tail_classes = head_classes
        # 3step, 20classes： 6-8-6
        for cls_idx in range(num_classes):
            if cls_idx < head_classes:
                step = 0
            elif head_classes <= cls_idx < num_classes - tail_classes:
                step = 1
            else:
                step = 2
            img_num = img_max * imb_factor**(step / 2)
            num_per_cls.append(int(img_num))
    else:
        num_per_cls.extend([int(img_max)] * num_classes)

    new_img_paths = []
    # new_targets = []

    img_paths = np.array(img_paths)
    for the_class, the_img_num in zip(classes, num_per_cls):
        # random shuffle indexs and select num_samples of the class
        indexes = np.where(targets == the_class)[0]
        np.random.shuffle(indexes)
        selec_idx = indexes[:the_img_num]
        # generate new train pairs
        new_img_paths.extend(img_paths[selec_idx].tolist())
        # new_targets.extend([the_class, ] * the_img_num)

    return new_img_paths


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    data_root = "/home/chenghua/Data/miniImageNet"
    img_lst_fpath = join(data_root, "train3.txt")

    img_paths = []
    targets = []
    with open(img_lst_fpath) as f:
        for line in f.readlines():
            img_path, label_str = line.split()
            img_paths.append(data_root + img_path)
            targets.append(int(label_str))

    LT_img_paths = get_LT_img_paths(img_paths, targets, 'exp', 0.02)

    train_mean, train_std = compute_mean_and_std(LT_img_paths)

    print('Done.')
