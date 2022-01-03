import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class CutMix(Dataset):
    """Referred to https://github.com/ildoonet/cutmix"""

    def __init__(self,
                 dataset,
                 num_mix=1,
                 beta=1.,
                 prob=1.0,
                 remix=False,
                 adapt=-1):
        """build cutmix dataset based on common custom dataset class.

        Args:
            dataset(Dataset object): common custom dataset classes
            num_mix(int): number of imgs to mix
            beta(float): drop a lambd
            prob(float): probability of using cutmix
        """
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.num_samples_per_cls = dataset.num_samples_per_cls
        self.class_weight = dataset.class_weight
        self.indexes_per_cls = dataset.indexes_per_cls

        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.remix = remix
        self.adapt = adapt

        # get all choices for each class
        self.choices = self.get_choices()  # chooice for each class
        self.class_pointers = [0] * self.num_classes

    def __getitem__(self, index):
        img, target = self.dataset[index]
        target_onehot = single_label2onehot(self.num_classes, target)

        for _ in range(self.num_mix):
            r = np.random.rand(1)

            if self.beta <= 0 or r > self.prob:
                continue

            if self.adapt in [1, 2]:
                # rebalance mixup
                rand_index = random.choice(self.choices[target])
            elif self.adapt == 3:  # random class
                rand_class = random.choice(
                    list(range(target, self.num_classes)))
                class_pointer = self.class_pointers[rand_class]
                self.class_pointers[rand_class] = \
                    (self.class_pointers[rand_class] + 1) % \
                    self.num_samples_per_cls[rand_class]
                class_indexes = self.indexes_per_cls[rand_class]
                rand_index = class_indexes[class_pointer]
            elif self.adapt == 4:  # reweight class
                rand = np.random.rand(1)
                rand_class = -1

                # see which interval it falls in.
                # self.class_weight: [0.01, 0.xx, 0.1, xxx, 0.4]
                # self.class_weight[:i+1]: [0,...,i]

                for i in range(self.num_classes):
                    if rand <= np.sum(self.class_weight[:i + 1]):
                        rand_class = i

                        break
                class_pointer = self.class_pointers[rand_class]
                self.class_pointers[rand_class] = \
                    (self.class_pointers[rand_class] + 1) % \
                    self.num_samples_per_cls[rand_class]
                class_indexes = self.indexes_per_cls[rand_class]
                rand_index = class_indexes[class_pointer]
            else:
                rand_index = random.choice(range(len(self)))

            img2, target2 = self.dataset[rand_index]
            target2_onehot = single_label2onehot(self.num_classes, target2)

            # generate mixed weight lambd for original area
            lambd = np.random.beta(self.beta, self.beta)

            if self.remix:  # the weight of choosen-to-mix img is larger.
                lambd = lambd if lambd < 0.5 else 1 - lambd

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lambd)  # set box
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]

            # 为什么要重新算weight？
            # 因为cut区域时，可能超出边界，所以实际的区域更小，故而要重新计算。
            cut_area = (bbx2 - bbx1) * (bby2 - bby1)
            whole_area = img.size(-1) * img.size(-2)  # H * W
            weight = 1 - cut_area / whole_area  # for original img

            target_onehot = target_onehot * weight + \
                target2_onehot * (1. - weight)

        return img, target_onehot

    def __len__(self):
        return len(self.dataset)

    def get_choices(self):
        targets = self.dataset.targets
        choices = []

        for i in range(self.num_classes):
            if self.adapt == 1:  # class i only select class i+k(k>0)
                if i == self.num_classes - 1:
                    choices.append(targets)

                    break
                indexes = np.where(np.array(targets) > i)[0].tolist()
            elif self.adapt == 2:  # class i select class i+k(k>=0)
                indexes = np.where(np.array(targets) >= i)[0].tolist()
            else:
                indexes = []
            choices.append(indexes)

        return choices


def single_label2onehot(size, target):
    # single label to one-hot
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.

    return vec


def rand_bbox(size, lambd):
    # size: (N, C, H, W)

    if len(size) in [3, 4]:
        H, W = size[-2], size[-1]
    else:
        raise Exception

    cut_ratio = np.sqrt(1. - lambd)
    cut_h = np.int(H * cut_ratio)
    cut_w = np.int(W * cut_ratio)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
