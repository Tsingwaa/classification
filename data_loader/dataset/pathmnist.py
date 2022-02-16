# To ensure fairness, we use the same code in
# LDAM (https://github.com/kaidic/LDAM-DRW) &
# BBN (https://github.com/Megvii-Nanjing/BBN)
# to produce long-tailed CIFAR datasets.

from os.path import join

import numpy as np
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from medmnist import PathMNIST
from PIL import Image
import random
# from pudb import set_trace


@Datasets.register_module("ImbalancedPathMNIST")
class ImbalancedPathMNIST(PathMNIST):
    num_classes = 9
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    def __init__(self,
                 phase,
                 root="medmnist",
                 transform=None,
                 download=False,
                 imb_type='exp',
                 imb_factor=0.01,
                 **kwargs):
        if '/' not in root:
            root = join(DATASETS_ROOT, root)

        super(ImbalancedPathMNIST, self).__init__(root=root,
                                                  split=phase,
                                                  download=download)
        self.phase = phase
        self.transform = transform
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        # self.class_adapt = kwargs.get('class_adapt', False)
        np.random.seed(0)

        self.data = self.imgs
        # 8: 12885 -> 0
        # 5: 12182 -> 1
        # 3: 10401 -> 2
        # 2: 10360 -> 3
        # 1: 9509  -> 4
        # 7: 9401  -> 5
        # 0: 9366  -> 6
        # 4: 8006  -> 7
        # 6: 7886  -> 8
        # remap = {
        #     8: 0,
        #     5: 1,
        #     3: 2,
        #     2: 3,
        #     1: 4,
        #     7: 5,
        #     0: 6,
        #     4: 7,
        #     6: 8,
        # }
        # 0: 94 -> 0
        # 1: 1  -> 1
        # 2: 86 -> 2
        # 3: 0  -> 8
        # 4: 45 -> 4
        # 5: 5  -> 7
        # 6: 45 -> 5
        # 7: 34 -> 6
        # 8: 72 -> 3

        remap = {
            0: 0,
            1: 1,
            2: 2,
            3: 8,
            4: 4,
            5: 7,
            6: 5,
            7: 6,
            8: 3,
        }

        self.targets = [
            remap[label] for label in self.labels.squeeze().tolist()
        ]
        # self.targets = self.labels.squeeze().tolist()

        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]

        if phase == 'train':
            max_num_samples = self.num_samples_per_cls[0]
            self.num_samples_per_cls = self.get_img_num_per_cls(
                max_num_samples)
            self.gen_imbalanced_data()

            # get other property
            self.class_weight = self.get_class_weight()
            self.indexes_per_cls = self.get_indexes_per_cls()

        self.group_mode = "class"

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
        img = Image.fromarray(img)

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

    def __len__(self):
        return len(self.targets)

@Datasets.register_module("ImbalancedPathMNIST_BBN")
class ImbalancedPathMNIST_BBN(PathMNIST):
    num_classes = 9
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    def __init__(self,
                 phase,
                 dual_sample=False,
                 dual_sample_type=None,
                 root="medmnist",
                 transform=None,
                 download=False,
                 imb_type='exp',
                 imb_factor=0.01,
                 **kwargs):
        if '/' not in root:
            root = join(DATASETS_ROOT, root)

        super(ImbalancedPathMNIST_BBN, self).__init__(root=root,
                                                  split=phase,
                                                  download=download)
        self.dual_sample = True if dual_sample and phase =='train' else False
        self.dual_sample_type = dual_sample_type
        self.phase = phase
        self.transform = transform
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        # self.class_adapt = kwargs.get('class_adapt', False)
        np.random.seed(0)

        self.data = self.imgs

        remap = {
            0: 0,
            1: 1,
            2: 2,
            3: 8,
            4: 4,
            5: 7,
            6: 5,
            7: 6,
            8: 3,
        }

        self.targets = [
            remap[label] for label in self.labels.squeeze().tolist()
        ]
        # self.targets = self.labels.squeeze().tolist()

        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]

        if phase == 'train':
            max_num_samples = self.num_samples_per_cls[0]
            self.num_samples_per_cls = self.get_img_num_per_cls(
                max_num_samples)
            self.gen_imbalanced_data()

            # get other property
            self.class_weight = self.get_class_weight()
            self.indexes_per_cls = self.get_indexes_per_cls()

        self.group_mode = "class"
        if self.dual_sample:
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.num_classes)
            self.class_dict = self._get_class_dict()

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
        img = Image.fromarray(img)

        meta = dict()
        if self.dual_sample:
            if self.dual_sample_type == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.dual_sample_type == "balance":
                sample_class = random.randint(0, self.num_classes-1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.dual_sample_type == "uniform":
                sample_index = random.randint(0, self.__len__() - 1)

            sample_img, sample_label = self.data[sample_index], self.targets[sample_index]
            sample_img = Image.fromarray(sample_img)
            sample_img = self.transform(sample_img)

            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, target, meta

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

    def __len__(self):
        return len(self.targets)

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

if __name__ == '__main__':
    trainset = ImbalancedPathMNIST(root="medmnist",
                                   phase="train",
                                   download=False,
                                   transform=None)
