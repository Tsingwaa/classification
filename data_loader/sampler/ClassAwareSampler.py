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
from data_loader.sampler.builder import Samplers


class RandomCycleIter:
    def __init__(self, data_list, test_mode=False):
        self.data_list = list(data_list)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        # 取下一个元素i + 1；初始时，为self.length
        if self.i == self.length:
            # 如果迭代完一轮，则从头来过；
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, total_samples,
                                 num_samples_each_cls_draw=1):
    i = 0
    j = 0
    while i < total_samples:
        if j >= num_samples_each_cls_draw:
            j = 0

        if j == 0:
            temp_tuple = next(zip(
                *[data_iter_list[next(cls_iter)]] *
                num_samples_each_cls_draw
            ))
            """解读上面一行代码:
            data_iter_list[next(cls_iter)]是选定类别的数据迭代器；
            先复制n个迭代器，然后解索引选定n个迭代器，然后zip组合为一起，
            再用next读取组合后的迭代器，依次读取n个同一位置的迭代器的下一元素，
            该轮循环的效果就是，并行取了n个该类的数据索引；

            最终的生成效果就是，总的抽取total_samples，每次抽取一个类别的n个数量；
            """
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


@Samplers.register_module('ClassAwareSampler')
class ClassAwareSampler (Sampler):

    def __init__(self, dataset, num_samples_each_cls_draw=1, **kwargs):
        """根据dataset得到类等价的sampler, 上采样器
        Args:
            dataset: 数据集对象，使用其labels对象
            num_samples_each_cls_draw: 每类每次抽取的样本数量
        """
        num_classes = len(np.unique(dataset.labels))
        self.class_iter = RandomCycleIter(range(num_classes))

        # turn list [0,..., num_classes-1] to RandomCycleIterator
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(dataset.labels):
            cls_data_list[label].append(i)

        # 每类样本的索引列表，生成各类index的Iterator
        self.data_iter_list = [
            RandomCycleIter(this_cls_data) for this_cls_data in cls_data_list
        ]

        # 最大采样数 * 类数， 其实就是以最大类为标准采样
        max_num_samples = max([len(x) for x in cls_data_list])
        self.num_samples = max_num_samples * len(cls_data_list)

        # 每类取的样本数量
        self.num_samples_each_cls_draw = num_samples_each_cls_draw

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter,
                                            self.data_iter_list,
                                            self.num_samples,
                                            self.num_samples_each_cls_draw)

    def __len__(self):
        return self.num_samples


def get_sampler():
    return ClassAwareSampler


def get_balanced_samper(dataset):
    # 基于dataset，设置各类的data indexes
    buckets = [[] for _ in range(dataset.cls_num)]
    for idx, label in enumerate(dataset.labels):
        buckets[label].append(idx)
    sampler = BalancedSampler(buckets, retain_epoch_size=True)
    return sampler


@Samplers.register_module('BalanceSampler')
class BalancedSampler(Sampler):
    def __init__(self, dataset, retain_epoch_size=True, **kwargs):
        self.num_buckets = len(np.unique(dataset.labels))
        self.buckets = [[] for _ in range(self.num_buckets)]
        for idx, label in enumerate(dataset.labels):
            self.buckets[label].append(idx)

        # Pointer for each class to mark which index we should choose next.
        # Initialize pointer as the first one (index = 0)
        self.bucket_pointers = [0 for _ in range(self.num_buckets)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        # Randomly choose one class
        bucket_idx = random.randint(0, self.num_buckets - 1)
        # get indexes of corresponding class
        bucket = self.buckets[bucket_idx]
        # choose the pointed item
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            # if all is drawn, shuffle and redraw.
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)

        return item

    def __len__(self):
        if self.retain_epoch_size:
            # Acrually we need to upscale to next full batch
            return sum([len(bucket) for bucket in self.buckets])
        else:
            # Ensures every instance has the chance to be visited in an epoch
            # oversample each class size to max class size.
            return max(
                [len(bucket) for bucket in self.buckets]
            ) * self.bucket_num
