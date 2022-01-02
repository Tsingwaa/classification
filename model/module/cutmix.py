import numpy as np
import random
import torch
from torch.utils.data.dataset import Dataset


class CutMix(Dataset):
    """Referred to https://github.com/ildoonet/cutmix"""

    def __init__(self, dataset, num_mix=1, beta=1., prob=1.0):
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

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = single_label2onehot(self.num_classes, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample, lambd for original area
            lambd = np.random.beta(self.beta, self.beta)

            # TODO add class-specific weight prob.
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = single_label2onehot(self.num_classes, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lambd)  # set box
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            # 为什么要重新算weight？
            # 因为cut区域时，可能超出边界，所以实际的区域更小，故而要重新计算。
            cut_area = (bbx2 - bbx1) * (bby2 - bby1)
            whole_area = img.size(-1) * img.size(-2)  # H * W
            weight = 1 - cut_area / whole_area  # for original img

            lb_onehot = lb_onehot * weight + lb2_onehot * (1. - weight)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


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
