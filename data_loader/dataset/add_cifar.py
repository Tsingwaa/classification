# To ensure fairness, we use the same code in
# LDAM (https://github.com/kaidic/LDAM-DRW) &
# BBN (https://github.com/Megvii-Nanjing/BBN)
# to produce long-tailed CIFAR datasets.
import os
import random

import numpy as np
import PIL
# import torch
import torchvision
# from pudb import set_trace
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from torchvision import transforms


@Datasets.register_module("Add_ImbalanceCIFAR10")
class Add_ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    def __init__(self,
                 root,
                 phase,
                 transform=None,
                 download=True,
                 imb_type='exp',
                 imb_factor=0.01,
                 extra_classes_num=0,
                 extra_samples_num=0,
                 split_type='same',
                 alpha=1,
                 **kwargs):
        train = True if phase == 'train' else False
        root = os.path.join(DATASETS_ROOT, root)
        super(Add_ImbalanceCIFAR10, self).__init__(root=root,
                                                   train=train,
                                                   transform=transform,
                                                   download=download)

        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.class_adapt = kwargs.get('class_adapt', False)
        np.random.seed(0)

        self.num_samples_per_cls = self.get_img_num_per_cls()

        self.gen_imbalanced_data()

        # get other property
        self.class_weight = self.get_class_weight()
        self.indexes_per_cls = self.get_indexes_per_cls()
        self.extra_classes_num = extra_classes_num
        self.num_of_samples_per_extra_class = \
            self.get_num_of_samples_per_extra_class(
                split_type, extra_samples_num)
        self.add_extra_classes()
        self.img_num = self.num_samples_per_cls +\
            self.num_of_samples_per_extra_class
        self.num_classes += extra_classes_num

    def get_img_num_per_cls(self):
        max_num_samples = int(len(self.data) / self.num_classes)
        num_samples_per_cls = []

        if self.imb_type == 'exp':
            for class_index in range(self.num_classes):
                num_samples =\
                    max_num_samples * (self.imb_factor ** (
                        class_index / (self.num_classes - 1.0)))
                num_samples_per_cls.append(int(num_samples))
        elif self.imb_type == 'step':
            # One step: the former half {img_max} imgs,
            # the latter half {img_max * imb_factor} imgs
            half_num_classes = int(self.num_classes // 2)

            for class_index in range(self.num_classes):
                if class_index <= half_num_classes:
                    num_samples = max_num_samples
                else:
                    num_samples = int(max_num_samples * self.imb_factor)

                num_samples_per_cls.append(num_samples)
        else:
            # Original balance CIFAR dataset.
            num_samples_per_cls = [max_num_samples] * self.num_classes

        return num_samples_per_cls

    def gen_imbalanced_data(self):
        new_data = []
        new_targets = []
        targets = np.array(self.targets, dtype=np.int64)
        class_indexes = np.unique(targets)
        # np.unique default output by increasing order. i.e. {class 0}: MAX.
        # np.random.shuffle(classes)
        self.cls2nsamples = dict()

        for class_index, num_samples in zip(class_indexes,
                                            self.num_samples_per_cls):
            self.cls2nsamples[class_index] = num_samples
            img_indexes = np.where(targets == class_index)[0]  # get index
            # Shuffle indexes for each class.
            np.random.shuffle(img_indexes)
            select_indexes = img_indexes[:num_samples]
            new_data.append(self.data[select_indexes, ...])
            new_targets.extend([class_index] * num_samples)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_num_of_samples_per_extra_class(self, split_type,
                                           extra_samples_num):
        num_of_samples_per_extra_class = []

        if split_type == 'same':
            for i in range(self.extra_classes_num):
                num_of_samples_per_extra_class.append(extra_samples_num //
                                                      self.extra_classes_num)

        return num_of_samples_per_extra_class

    def add_extra_classes(self):

        lambd = 0.5
        c1 = self.num_classes - 1
        c2 = self.num_classes - 2
        extra_data = []
        extra_targets = []

        for i in range(self.extra_classes_num):
            for j in range(self.num_of_samples_per_extra_class[i]):
                index1 = random.choice(self.indexes_per_cls[c1])
                index2 = random.choice(self.indexes_per_cls[c2])
                img = lambd * self.data[index1] + (1 -
                                                   lambd) * self.data[index2]
                extra_data.append(img)
                extra_targets.append(self.num_classes + i)

            if c2 == 0:
                c1 = c1 - 1
                c2 = c1 - 1
            else:
                c2 = c2 - 1

        if self.extra_classes_num > 0:
            extra_data = np.array(extra_data, dtype=np.uint8)
            # extra_targets = np.array(extra_targets)
            print(extra_data.shape)
            # print(extra_targets.shape)
            print(self.data.shape)
            # print(self.targets.shape)
            self.data = np.concatenate((self.data, extra_data), axis=0)
            self.targets += extra_targets
            # self.targets = np.concatenate(self.targets, )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = PIL.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, target

    def get_class_weight(self):
        num_samples_per_cls = np.array(self.num_samples_per_cls)
        num_samples = np.sum(num_samples_per_cls)
        weight = num_samples / (self.num_classes * num_samples_per_cls)
        weight /= np.sum(weight)

        return weight

    def get_indexes_per_cls(self):
        indexes_per_cls = []

        for i in range(self.num_classes):
            indexes = np.where(np.array(self.targets) == i)[0].tolist()
            indexes_per_cls.append(indexes)

        return indexes_per_cls


@Datasets.register_module("Add_ImbalanceCIFAR100")
class Add_ImbalanceCIFAR100(Add_ImbalanceCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    num_classes = 100
    cls_num = 100


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = Add_ImbalanceCIFAR100(root='./data',
                                     train=True,
                                     download=True,
                                     transform=transform)
    # trainloader = iter(trainset)
    # data, label = next(trainloader)
    # import pdb
    # pdb.set_trace()
