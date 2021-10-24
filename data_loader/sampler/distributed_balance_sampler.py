from torch.utils.data.sampler import Sampler
# from torch.utils.data import Dataset
# from torch.utils.data.distributed import DistributedSampler
from data_loader.sampler.builder import Samplers
from collections import defaultdict
# from tqdm import tqdm

import copy
import random
import math
import torch.distributed as dist
import numpy as np


@Samplers.register_module("DistributedRandomCategorySampler")
class DistributedRandomCategorySampler(Sampler):
    """
    Randomly sample N categories, then for each category,
    random sample K instances, therefore batch size is N*K.
    Args:
    - labels (list): dataset labels.
    - num_instances (int): number of instances per category in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, labels, batch_size, num_instances=8,
                 num_replicas=None, rank=None, manual_seed=0):
        self.labels = labels
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_labels_per_batch = self.batch_size // self.num_instances
        self.label2idxs = defaultdict(list)
        self.manual_seed = manual_seed
        self.epoch = 1

        # {label1:[index1, index2,...], ...}
        for index, label in enumerate(self.labels):
            self.label2idxs[label].append(index)
        self.unique_labels = list(self.label2idxs.keys())

        # estimate number of examples in an epoch
        self.total_length = 0
        for label in self.unique_labels:
            idxs = self.label2idxs[label]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.total_length += num - num % self.num_instances

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()

        self.length = math.ceil(self.total_length / num_replicas)
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        # Generate batch for each identity
        for label in self.unique_labels:
            idxs = copy.deepcopy(self.label2idxs[label])
            if len(idxs) < self.num_instances:  # 对于数量不足的类，自动重复采样replace=True
                np.random.seed(self.manual_seed + self.epoch)
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True)
            random.seed(self.manual_seed + self.epoch)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[label].append(batch_idxs)  # 作为一组batch
                    batch_idxs = []

        # Generate final idxs
        avai_labels = copy.deepcopy(self.unique_labels)
        final_idxs = []
        while len(avai_labels) >= self.num_labels_per_batch:
            random.seed(self.manual_seed + self.epoch)
            selected_labels = random.sample(
                avai_labels, self.num_labels_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)

        # split idxs
        local_num = int(len(final_idxs) / self.num_replicas)
        local_indices = final_idxs[local_num *
                                   self.rank: local_num * (self.rank + 1)]
        print(f"Generated {local_num} index at rank {self.rank}"
              f" by DistributedRandomCategorySampler.")
        return iter(local_indices)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch


@Samplers.register_module("DistributedRandomGroupSampler")
class DistributedRandomGroupSampler(Sampler):
    """
    Seperate data of each category into integral groups, and random sample
    groups to create batches for each iterations.

    Args:
    - labels (list): dataset labels.
    - num_instances (int): number of instances per category in a batch.
    - batch_size (int): number of samples per batch
    """

    def __init__(self, labels, batch_size, num_instances=8,
                 num_replicas=None, rank=None, manual_seed=0):
        self.labels = labels
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.label2idxs = defaultdict(list)
        self.manual_seed = manual_seed
        self.epoch = 1

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
            print(
                f"Initialing DistributedRandomGroupSampler at rank {rank}...")

        # {label1:[index1, index2,...], ...}
        for index, label in enumerate(self.labels):
            self.label2idxs[label].append(index)
        # all available label list
        self.unique_labels = list(self.label2idxs.keys())

        # estimate number of examples in an epoch
        self.total_groups = 0
        for label in self.unique_labels:
            idxs = self.label2idxs[label]
            groups = math.ceil(len(idxs) / num_instances)
            self.total_groups += groups

        self.length = math.floor(
            self.total_groups * 1.0 / num_replicas) * num_instances
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):

        # Generate groups (each group contains num_instances samples)
        group_list = []
        for label in self.unique_labels:
            idxs = self.label2idxs[label]
            group_idxs = []
            for idx in idxs:
                group_idxs.append(idx)
                if len(group_idxs) == self.num_instances:
                    group_list.append(group_idxs)
                    group_idxs = []

            # if there is remaining idxs, sample the needed num
            # from this category idxs.
            remaining_idxs_num = len(group_idxs)
            if remaining_idxs_num != 0:
                need_idxs_num = self.num_instances - remaining_idxs_num
                # random sample needed idxs from all alternatives.
                np.random.seed(self.manual_seed + self.epoch)
                need_idxs = np.random.choice(
                    idxs, size=need_idxs_num, replace=True)
                group_idxs.extend(need_idxs)
                group_list.append(group_idxs)

        # Shuffle groups
        random.seed(self.manual_seed + self.epoch)
        shuffled_group_list = random.shuffle(group_list)

        # Generate final idxs
        groups_per_batch = self.batch_size / self.num_instances

        # split idxs
        local_group_num = math.floor(len(group_list) / self.num_replicas)
        local_groups = group_list[local_group_num *
                                  self.rank: local_group_num * (self.rank + 1)]
        local_indices = []
        for group in local_groups:
            local_indices.extend(group)
        print(f"Generated {len(local_indices)} index at rank {self.rank}"\
              f" by DistributedRandomGroupSampler.")

        return iter(local_indices)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch
