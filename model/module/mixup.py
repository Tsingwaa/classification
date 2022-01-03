import numpy as np
import random
import torch
from torch.utils.data.dataset import Dataset


class MixUp(Dataset):
    """Referred to https://github.com/ildoonet/cutmix"""

    def __init__(self, dataset, num_mix=1, beta=1., prob=1.0, type='random'):
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
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.type = type

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = single_label2onehot(self.num_classes, lb)
        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue
            lambd = np.random.beta(self.beta, self.beta)
            
            img2, lb2_onehot = self.get_img2(lb)
            img = lambd * img + (1 - lambd) * img2
            lb_onehot = lb_onehot * lambd + lb2_onehot * (1. - lambd)

        return img, lb_onehot

    def get_img2(self, lb):
        if self.type == 'random':
            return self.get_img2_random(lb)
        elif self.type == 'A':
            return self.get_img2_A(lb)
        elif self.type == 'B':
            return self.get_img2_B(lb)
        else:
            return None, None

    def get_img2_A(self, lb):
        # generate mixed sample, lambd for original area
        rand_index = random.choice(range(len(self)))
        img2, lb2 = self.dataset[rand_index]
        while lb2 > lb:
            # print('lb: %d, lb2: %d'%(lb, lb2))
            rand_index = random.choice(range(len(self)))
            img2, lb2 = self.dataset[rand_index]
        lb2_onehot = single_label2onehot(self.num_classes, lb2)
        return img2, lb2_onehot

    def get_img2_B(self, lb):
        # generate mixed sample, lambd for original area
        rand_index = random.choice(range(len(self)))
        img2, lb2 = self.dataset[rand_index]
        while lb2 < lb:
            # print('lb: %d, lb2: %d'%(lb, lb2))
            rand_index = random.choice(range(len(self)))
            img2, lb2 = self.dataset[rand_index]
        lb2_onehot = single_label2onehot(self.num_classes, lb2)
        return img2, lb2_onehot

    def get_img2_random(self, lb):
        # generate mixed sample, lambd for original area
        rand_index = random.choice(range(len(self)))
        img2, lb2 = self.dataset[rand_index]
        lb2_onehot = single_label2onehot(self.num_classes, lb2)

        return img2, lb2_onehot

    def __len__(self):
        return len(self.dataset)



def single_label2onehot(size, target):
    # single label to one-hot
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec
