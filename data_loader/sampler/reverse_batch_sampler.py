import numpy as np
from PIL import Image

from torch.utils.data.sampler import BatchSampler

import random

@Sampler.register_module("reversedbatchsampler")
class ReversedBatchSampler(BatchSampler):

    def __init__(self, labels, batch_size, random_seed=42):
        """Reverse batch sampler for imbalanced dataset.
        The probability of sampling a data point from the minority classes is higher than that
        from the majority classes.

        The detailed formula of computing sampling probability from class i:
            p_i = \frac{1}{n_i} / \sigma_j \frac{1}{n_j} 
        For numerical stability and precision, the optimized formula is as followed:
            p_i = \frac{n_max}{n_i} / \sigma_j \frac{n_max}{n_j}
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        random.seed(random_seed)

        self.labels_set = list(set(self.labels))
        self.label_to_indices = {
            label: np.where(self.labels == label)[0]
            for label in self.labels_set
            }

        self.cls_num = len(self.labels_set)
        self.num_list = [len(self.label_to_indices[label])
                         for label in self.labels_set]
        max_num = max(self.num_list)
        self.class_weight = [max_num / i for i in self.num_list]
        self.sum_weight = sum(self.class_weight)

    # from BBN
    # https://github.com/Megvii-Nanjing/BBN/blob/7992e908842f5934f0d1ee3f430d796621e81975/lib/dataset/imbalance_cifar.py#L59
    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def __iter__(self):
        indices = []
        for i in range(self.batch_size):
            sample_class = self.sample_class_index_by_weight()
            sample_index = random.choice(self.label_to_indices[sample_class])
            indices.append(sample_index)

        return indices

    def __len__(self):
        return len(self.labels) // self.batch_size

