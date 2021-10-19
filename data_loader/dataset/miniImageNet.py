"""Adopted from https://github.com/Megvii-Nanjing/BBN
Customized by Kaihua Tang
"""

import os
import torch
import numpy as np
from os.path import join
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from data_loader.dataset.builder import Datasets


@Datasets.register_module('imb_miniImageNet')
class ImbalanceMiniImageNet(torch.utils.data.Dataset):
    cls_num = 100

    def __init__(self, root, phase, imbalance_ratio, imb_type='exp',
                 transform=None, **kwargs):
        self.img_path = []
        self.targets = []
        img_lst_fpath = os.path.join(root, phase + '.txt')
        with open(img_lst_fpath) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0][1:]))
                self.targets.append(int(line.split()[1]))

        self.img_path = np.array(self.img_path)
        self.img_num = []

        if phase == 'train':
            img_num_list = self.get_img_num_per_cls(
                self.cls_num, imb_type, imbalance_ratio
            )
            self.gen_imbalanced_data(img_num_list)
            self.img_num = img_num_list

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
            ])

        self.labels = self.targets

        print("{} Mode: Contain {} images".format(phase, len(self.img_path)))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if cat_id not in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.img_path) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_img_path = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_img_path.extend(self.img_path[selec_idx].tolist())
            new_targets.extend([the_class, ] * the_img_num)
        self.img_path = new_img_path
        self.targets = new_targets

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index  # , origin

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):

        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def get_category(self):
        label_dict = dict()
        for i in range(len(self.labels)):
            if self.labels[i] not in label_dict:
                category = self.img_path[i].split('/')[-2]
                label_dict[self.labels[i]] = category

        return label_dict


@Datasets.register_module("miniImageNet")
class miniImageNet(ImageFolder):
    # Full miniImageNet
    mean = [0.4759, 0.4586, 0.4153]
    std = [0.2847, 0.2785, 0.2955]

    def __init__(self, data_root, phase, transform=None, **kwargs):
        super(miniImageNet, self).__init__(
            root=join(data_root, phase),
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


@Datasets.register_module("miniImageNet_eval")
class miniImageNet_eval(ImageFolder):
    # Full miniImageNet
    mean = [0.4759, 0.4586, 0.4153]
    std = [0.2847, 0.2785, 0.2955]

    def __init__(self, data_root, phase, transform=None, **kwarg):
        super(miniImageNet_eval, self).__init__(
            root=join(data_root, phase),
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
