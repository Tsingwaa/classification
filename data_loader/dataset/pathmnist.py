# To ensure fairness, we use the same code in
# LDAM (https://github.com/kaidic/LDAM-DRW) &
# BBN (https://github.com/Megvii-Nanjing/BBN)
# to produce long-tailed CIFAR datasets.

from os.path import join

import numpy as np
import PIL
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from medmnist import PathMNIST


@Datasets.register_module("ImbPathMNIST")
class ImbalancePathMNIST(PathMNIST):
    num_classes = 7
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    def __init__(self,
                 root,
                 phase,
                 transform=None,
                 download=False,
                 imb_type='exp',
                 imb_factor=0.05,
                 **kwargs):
        if '/' not in root:
            root = join(DATASETS_ROOT, root)
        super(ImbalancePathMNIST, self).__init__(root=root,
                                                 split=phase,
                                                 transform=transform,
                                                 download=download)
        self.phase = phase
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        # self.class_adapt = kwargs.get('class_adapt', False)
        np.random.seed(0)

        self.data = self.imgs
        # 5: 12182 -> 0
        # 3: 10401 -> 1
        # 2: 10360 -> 2
        # 1: 9509  -> 3
        # 0: 9366  -> 4
        # 4: 8006  -> 5
        # 6: 7886  -> 6
        remap = {
            5: 0,
            3: 1,
            2: 2,
            1: 3,
            0: 4,
            4: 5,
            6: 6,
        }
        self.targets = [remap[label] for label in self.labels]

        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]
        max_num_samples = max(self.num_samples_per_cls)
        self.num_samples_per_cls = self.get_img_num_per_cls(max_num_samples)
        self.gen_imbalanced_data()

        # get other property
        self.class_weight = self.get_class_weight()
        self.indexes_per_cls = self.get_indexes_per_cls()
        self.group_mode = "shot"

    def get_img_num_per_cls(self, max_num_samples):
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
            # Original dataset.
            num_samples_per_cls = self.num_samples_per_cls

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


if __name__ == '__main__':
    trainset = ImbalancePathMNIST(root="medmnist",
                                  phase="train",
                                  download=False,
                                  transform=None)
