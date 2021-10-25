import random
import numpy as np
# from PIL import Image
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from common.dataset.sampler.builder import Sampler


@Sampler.register_module("balancedbatchsampler")
class BalancedBatchSampler(BatchSampler):
    """Balanced batch sampler for imbalance dataset.
    The probability of sampling data point from different class is equal.
    """

    def __init__(self, labels, batch_size, random_seed=42):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        random.seed(random_seed)

        self.labels_set = list(set(self.labels))
        self.label_to_indices = {
            label: np.where(self.labels == label)[0]
            for label in self.labels_set}
        self.cls_num = len(self.labels_set)

    def __iter__(self):
        indices = []
        for i in range(self.batch_size):
            sample_class = random.randint(0, self.cls_num - 1)
            sample_index = random.choice(self.label_to_indices[sample_class])
            indices.append(sample_index)

        return indices

    def __len__(self):
        return len(self.labels) // self.batch_size


@Sampler.register_module("balancedbatchsampler2")
class BalancedBatchSampler2(BatchSampler):
    """BatchSampler - from a MNIST-like dataset, samples n_classes and within
    these classes samples n_samples.

    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[
            0] for label in self.labels_set}
        for label in self.labels_set:
            np.random.shuffle(self.label_to_indices[label])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            self.labels_set = list(set(self.labels_set) - set(classes))
            if len(self.labels_set) < self.n_classes:
                self.labels_set = list(set(self.labels.numpy()))
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])  # noqa
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + \
                        self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


# if __name__ == '__main__':
#     train_batch_sampler = BalancedBatchSampler(train_dataset.labels, 128)
#     train_loader = DataLoader(train_dataset,
#                               batch_sampler=train_batch_sampler,
#                               num_workers=config.workers,
#                               pin_memory=True)
