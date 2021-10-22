"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import random

import numpy as np
from torch.utils.data.sampler import Sampler


##################################################################
# Class-aware sampling, partly implemented by frombeijingwithlove
##################################################################

class RandomCycleIter:
    """Accept a data_list and then return a Iterator which cyclely
    shuffles the data_list if in the test mode.

    Args:
        data_list: source list
        test_mode: decide whether to shuffle the list
    """

    def __init__(self, data_list, test_mode=False):
        self.data_list = list(data_list)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        # 第一次next, 默认i=length，则自动变为0
        if self.i == self.length:
            # 如果迭代完一轮，则从头来过；
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter,
                                 data_iter_list,
                                 num_samples,
                                 num_samples_cls=1,
                                 epoch=0,):
    """Generator: yield a list
    Args:
        cls_iter: (classes Iterator) Choose which classes to sample
        data_iter_list: (List of classes data Iterator)
        num_samples: how many samples in total for single generation.
        num_samples_cls:
        epoch:  set different seed
    """
    i = 0
    j = 0
    while i < num_samples:
        # yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(
                zip(*[
                    data_iter_list[next(cls_iter)]
                ] * num_samples_cls)
            )
            yield temp_tuple[j]

        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):
    def __init__(self, dataset, num_samples_cls=1,):
        """根据dataset得到类等价的sampler
        Args:
            dataset: 数据集对象，使用其labels对象
            num_samples_cls:
        """
        super(ClassAwareSampler, self).__init__(dataset)
        self.epoch = 0

        num_classes = len(np.unique(dataset.labels))

        # turn list [0,..., num_classes-1] to RandomCycleIterator
        self.class_iter = RandomCycleIter(range(num_classes))

        # num_classes * []: 二维列表，按类别收集对应索引
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(dataset.labels):
            cls_data_list[label].append(i)

        # 对每类样本的索引列表，生成Iterator
        self.data_iter_list = [
            RandomCycleIter(data_this_cls) for data_this_cls in cls_data_list
        ]

        # 最大采样数 * 类数， 以最大类的数量为标准采样
        max_num_samples = max([len(x) for x in cls_data_list])
        self.num_samples = max_num_samples * len(cls_data_list)

        # TODO what does num_samples_cls means???
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(
            self.class_iter,
            self.data_iter_list,
            self.num_samples,
            self.num_samples_cls,
            self.epoch
            )

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
