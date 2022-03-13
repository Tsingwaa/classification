# import math
import random

import numpy as np
import torch
# from pudb import set_trace
# from torch import serialization
from torch.utils.data.dataset import Dataset


class CutMix(Dataset):
    """Referred to https://github.com/ildoonet/cutmix"""

    def __init__(
        self,
        dataset,
        beta=1.,
        adapt_type=-1,
        remix_type=-1,
        kappa=3,
        mu=0.4,
        **kwargs,
    ):
        """build CutMix Dataset based on common custom dataset class.

        Args:
            dataset(Dataset object): common custom dataset classes
            num_mix: number of imgs to mix
            beta: drop a lambd
            prob: probability of using cutmix
            remix: referred to rebalanced mixup
            adapt: code for rebalancing adapt strategy
        """
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.phase = dataset.phase
        self.num_samples_per_cls = dataset.num_samples_per_cls
        # self.class_weight = dataset.class_weight
        # self.indexes_per_cls = dataset.indexes_per_cls
        # self.phase = self.dataset.phase
        self.beta = beta

        self.adapt_type = adapt_type
        self.remix_type = remix_type
        self.kappa = kappa
        self.mu = mu
        # get all choices for each class
        self.choices = self.get_choices()  # chooice for each class
        # self.class_pointers = [0] * self.num_classes

    def __getitem__(self, index):
        img, target = self.dataset[index]
        target_onehot = single_label2onehot(self.num_classes, target)

        if self.adapt_type == 2:
            rand_index = random.choice(self.choices[target])
        # if self.adapt in [1, 2, 8]:
        #     rand_index = random.choice(self.choices[target])
        # elif self.adapt == 3:  # random class
        #     rand_class = random.choice(
        #         list(range(target, self.num_classes)))
        #     candidate_indexes = self.indexes_per_cls[rand_class]
        #     rand_index = random.choice(candidate_indexes)
        # elif self.adapt == 4:  # inverse weight select class
        #     rand_class = random.choices(
        #         range(target, self.num_classes),
        #         weights=self.class_weight[target:])[0]
        #     candidate_indexes = self.indexes_per_cls[rand_class]
        #     rand_index = random.choice(candidate_indexes)
        # elif self.adapt == 5:  # weight: 大类更大
        #     weights = scale_(self.num_samples_per_cls, k=2)
        #     rand_class = random.choices(range(target, self.num_classes),
        #                                 weights=weights[target:])[0]
        #     candidate_indexes = self.indexes_per_cls[rand_class]
        #     rand_index = random.choice(candidate_indexes)
        # elif self.adapt == 6:  # 等可能选两边的类
        #     select_num_classes = math.log10(self.num_classes)
        #     floor_class = math.floor(target - 1 / 2 * select_num_classes)
        #     ceil_class = math.ceil(target + 1 / 2 * select_num_classes)

        #     if floor_class < 0:
        #         floor_class = 0
        #         ceil_class = select_num_classes

        #     if ceil_class >= self.num_classes:
        #         floor_class = self.num_classes - select_num_classes - 1
        #         ceil_class = self.num_classes - 1

        #     rand_class = np.random.randint(floor_class, ceil_class + 1)
        #     candidate_indexes = self.indexes_per_cls[rand_class]
        #     rand_index = random.choice(candidate_indexes)
        # elif self.adapt == 7:
        #     rand_class = random.choice(list(range(self.num_classes)))
        #     candidate_indexes = self.indexes_per_cls[rand_class]
        #     rand_index = random.choice(candidate_indexes)
        else:
            rand_index = random.choice(range(len(self)))

        img2, target2 = self.dataset[rand_index]
        target2_onehot = single_label2onehot(self.num_classes, target2)

        # generate mixed weight lambd for original area
        lambda_x = np.random.beta(self.beta, self.beta)

        # while lambda_x < self.lambd_thres:
        #     lambda_x = np.random.beta(self.beta, self.beta)

        # CutMix Image img---(lambda_xo) img2---(1-lambda_xo)
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lambda_x)
        img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]

        # 为什么要重新算weight？
        # 因为cut区域时，可能超出边界，所以实际的区域可能更小，故而要重新计算。
        cut_area = (bbx2 - bbx1) * (bby2 - bby1)
        whole_area = img.size(-1) * img.size(-2)  # H * W
        lambda_y2 = cut_area / whole_area  # for img2 and label2

        if self.remix_type == 0:  # Directly use kappa=3 and tau=0.5
            n1 = self.num_samples_per_cls[target]
            n2 = self.num_samples_per_cls[target2]

            if n1 >= self.kappa * n2 and lambda_y2 > 0.5:
                lambda_y2 = 1
            elif n1 * self.kappa <= n2 and lambda_y2 < 0.5:
                lambda_y2 = 0

        elif self.remix_type == 1:
            n1 = self.num_samples_per_cls[target]
            n2 = self.num_samples_per_cls[target2]

            if n1 >= self.kappa * n2 and lambda_y2 > 0.5:
                lambda_y2 = 1
            elif n1 >= self.kappa * n2 and self.mu <= lambda_y2 <= 0.5:
                lambda_y2 = 1 - lambda_y2

        # elif self.remix_v2:
        #     n1 = self.num_samples_per_cls[target]
        #     n2 = self.num_samples_per_cls[target2]

        #     if n1 >= self.kappa * n2 and lambda_y2 > 0.5:
        #         lambda_y2 = 1
        #     elif lambda_y2 < 0.5:
        #         lambda_y2 = 0
        #     else:
        #         lambda_y2 = min(lambda_y2, 1 - lambda_y2)
        # elif self.remix_v3:
        #     tau = 0.5
        #     n1 = self.num_samples_per_cls[target]
        #     n2 = self.num_samples_per_cls[target2]

        #     if n1 >= self.kappa * n2 and lambda_yo < tau:
        #         lambda_yo = 0
        #     elif self.tau2 <= lambda_yo <= 1 - self.tau2:
        #         lambda_yo = min(lambda_yo, 1 - lambda_yo)

        # elif self.soft_remix:
        #     lambda_yo = min(lambda_yo, 1 - lambda_yo)

        target_onehot = target2_onehot * lambda_y2 + \
            target_onehot * (1. - lambda_y2)

        return img, target_onehot

    def __len__(self):
        return len(self.dataset)

    def get_choices(self):
        """输出一个列表，这个列表含有各个背景类别条件下的前景类备选样本"""
        targets = self.dataset.targets
        choices = []

        for i in range(self.num_classes):
            if self.adapt_type == 1:  # class i only select class i+k(k>0)
                if i == self.num_classes - 1:
                    choices.append(targets)

                    break
                indexes = np.where(np.array(targets) > i)[0].tolist()
            elif self.adapt_type == 2:  # class i select class i+k(k>=0)
                indexes = np.where(np.array(targets) >= i)[0].tolist()
            elif self.adapt_type == 8:
                indexes = np.where(np.array(targets) <= i)[0].tolist()
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


def scale_(a: np.ndarray, k: int):
    a_ = np.power(a, k)

    return a_ / sum(a_)
