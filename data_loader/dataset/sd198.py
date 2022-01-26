# import os
from os.path import join

# import cv2
import numpy as np
import pandas as pd
import torch
# import torchvision.transforms as T
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from PIL import Image


@Datasets.register_module("SD198")
class SD198(torch.utils.data.Dataset):
    num_classes = 198
    splitfold_mean_std = [([0.4511, 0.4789, 0.5920], [0.2451, 0.2437, 0.2659]),
                          ([0.4536, 0.4828, 0.5990], [0.2458, 0.2443, 0.2655]),
                          ([0.4534, 0.4825, 0.5981], [0.2448, 0.2438, 0.2665]),
                          ([0.4520, 0.4799, 0.5938], [0.2460, 0.2443, 0.2652]),
                          ([0.4564, 0.4862, 0.6042], [0.2455, 0.2445, 0.2658])]

    def __init__(self, root, phase, fold_i=0, transform=None, **kwargs):
        if "/" not in root:
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.fold_i = fold_i
        self.transform = transform

        self.img_names, self.targets = self.get_data(fold_i, root, phase)

        data_dir = join(root, 'images')
        self.img_paths = [
            join(data_dir, img_name) for img_name in self.img_names
        ]

        class_idx_path = join(root, 'class_idx.npy')
        class_name_idx = np.load(class_idx_path)
        self.class_to_idx = {
            cls_idx: cls_name

            for cls_name, cls_idx in class_name_idx
        }
        self.classes = list(self.class_to_idx.values())

        self.num_samples_per_cls = [
            self.targets.count(cls_i) for cls_i in range(self.num_classes)
        ]
        self.group_mode = "class"

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            mean, std = self.splitfold_mean_std[self.fold_i]
            img = self.transform(img, mean=mean, std=std)

        return img, target

    def __len__(self):
        return len(self.targets)

    def get_data(self, fold_i, root, phase):
        img_lst_fpath = join(root, f"8_2_split/{phase}_{fold_i}.txt")
        df = pd.read_csv(img_lst_fpath, sep=" ")
        img_names = [d[0] for d in df.values]
        targets = [d[1] for d in df.values]

        return img_names, targets


if __name__ == '__main__':
    trainset = SD198(phase='train', fold_i=0)
