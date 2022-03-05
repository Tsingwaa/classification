import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class MixUp(Dataset):
    """Referred to https://github.com/ildoonet/cutmix"""

    def __init__(self,
                 dataset,
                 num_mix=1,
                 beta=1.,
                 prob=1.0,
                 mode='random',
                 remix=False):
        """build cutmix dataset based on common custom dataset class.

        Args:
            dataset(Dataset object): common custom dataset classes
            num_mix(int): number of imgs to mix
            beta(float): drop a lambd
            prob(float): probability of using cutmix
        """
        self.dataset = dataset
        self.phase = self.dataset.phase
        self.num_classes = dataset.num_classes
        self.num_samples_per_cls = dataset.num_samples_per_cls
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.mode = mode
        self.remix = remix
        self.idx_per_class = self.get_idx_per_class()

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = single_label2onehot(self.num_classes, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)

            if self.beta <= 0 or r > self.prob:
                continue
            lambd = np.random.beta(self.beta, self.beta)
            img2, lb2 = self.get_img2(lb)
            lb2_onehot = single_label2onehot(self.num_classes, lb2)

            if self.remix:  # Directly use kappa=3 and tau=0.5
                kappa, tau = 3, 0.5
                n1 = self.num_samples_per_cls[lb]
                n2 = self.num_samples_per_cls[lb2]

                if n1 >= kappa * n2 and lambd < tau:
                    lambd = 0
                elif n1 * kappa <= n2 and lambd > 1 - tau:
                    lambd = 1

            img = lambd * img + (1 - lambd) * img2
            lb_onehot = lb_onehot * lambd + lb2_onehot * (1. - lambd)

        return img, lb_onehot

    def get_img2(self, label):
        if self.mode == 'random':
            return self.get_img2_random(label)
        elif self.mode == 'A':
            return self.get_img2_A(label)
        elif self.mode == 'B':
            return self.get_img2_B(label)
        elif self.mode == 'C':
            return self.get_img2_C(label)
        elif self.mode == 'D':
            return self.get_img2_D(label)
        elif self.mode == 'E':
            return self.get_img2_E(label)
        else:
            raise ValueError

    def get_img2_A(self, lb):
        rand_index = random.choice(range(len(self)))
        img2, lb2 = self.dataset[rand_index]

        while lb2 > lb:
            # print('lb: %d, lb2: %d'%(lb, lb2))
            rand_index = random.choice(range(len(self)))
            img2, lb2 = self.dataset[rand_index]

        return img2, lb2

    def get_img2_B(self, lb):
        rand_index = random.choice(range(len(self)))
        img2, lb2 = self.dataset[rand_index]

        while lb2 < lb:
            # print('lb: %d, lb2: %d'%(lb, lb2))
            rand_index = random.choice(range(len(self)))
            img2, lb2 = self.dataset[rand_index]

        return img2, lb2

    def get_img2_C(self, lb):
        weight = np.power(self.num_samples_per_cls, 2)
        class_idx = np.random.choice(range(self.num_classes), p=weight)
        rand_index = np.random.choice(self.idx_per_class[class_idx])
        img2, lb2 = self.dataset[rand_index]

        return img2, lb2

    def get_img2_D(self, lb):
        rand_index = random.choice(range(len(self)))
        img2, lb2 = self.dataset[rand_index]

        while abs(lb - lb2) > np.log(self.num_classes):
            # print('lb: %d, lb2: %d'%(lb, lb2))
            rand_index = random.choice(range(len(self)))
            img2, lb2 = self.dataset[rand_index]

        return img2, lb2

    def get_img2_E(self, lb):
        rand_index = random.choice(range(len(self.idx_per_class[lb])))
        img2, lb2 = self.dataset[rand_index]

        return img2, lb2

    def get_img2_random(self, lb):
        rand_index = random.choice(range(len(self)))
        img2, lb2 = self.dataset[rand_index]

        return img2, lb2

    def __len__(self):
        return len(self.dataset)

    def get_idx_per_class(self):
        idx_per_class = [[] for i in range(self.num_classes)]

        for i in range(len(self.dataset)):
            _, lb = self.dataset[i]
            idx_per_class[lb].append(i)

        return idx_per_class


def single_label2onehot(size, target):
    # single label to one-hot
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.

    return vec
