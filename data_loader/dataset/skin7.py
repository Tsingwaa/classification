# import os
from os.path import join

import cv2
import pandas as pd
import torch
from data_loader.dataset.builder import DATASETS_ROOT, Datasets

# from PIL import Image


@Datasets.register_module("Skin7")
class Skin7(torch.utils.data.Dataset):
    """Original image size: (3, 450, 600)"""
    num_classes = 7
    splitfold_mean_std = [([0.5709, 0.5464, 0.7634], [0.1701, 0.1528, 0.1408]),
                          ([0.5707, 0.5462, 0.7634], [0.1707, 0.1531, 0.1416]),
                          ([0.5704, 0.5462, 0.7642], [0.1704, 0.1528, 0.1410]),
                          ([0.5701, 0.5458, 0.7636], [0.1702, 0.1530, 0.1413]),
                          ([0.5705, 0.5461, 0.7629], [0.1703, 0.1528, 0.1413])]

    def __init__(self, root, train, fold_i=0, transform=None, **kwargs):
        """fold_i: [0, 1, 2, 3, 4]"""

        self.train = train
        phase = "train" if train else "test"
        self.transform = transform
        self.mean, self.std = self.splitfold_mean_std[fold_i]

        if "/" not in root:
            root = join(DATASETS_ROOT, root)
        data_dir = join(root, 'ISIC2018_Task3_Training_Input')

        self.img_names, self.targets = self.get_data(fold_i, root, phase)

        self.img_paths = [
            join(data_dir, img_name) for img_name in self.img_names
        ]
        # class-indexes -> cardinality [223, 1341, 103, 66, 220, 23, 29]
        # the sorted cardinality       [1341, 223, 220, 103, 66, 29, 23]
        # sort the target by cardinality in decreasing order
        remap = {
            0: 1,
            1: 0,
            2: 3,
            3: 4,
            4: 2,
            5: 6,
            6: 5,
        }
        self.targets = [remap[target] for target in self.targets]

        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]

        print(f"===> Initialized fold-{fold_i} {phase}set")

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]
        # img = Image.open(img_path).convert('RGB')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, target

    def __len__(self):
        return len(self.targets)

    def get_data(self, fold_i, root, phase):
        """fetch the img names and targets"""

        fold_i += 1
        csv_fpath = join(root, "split_data", "origin_split_data",
                         f"split_data_{fold_i}_fold_{phase}.csv")
        dataframe = pd.read_csv(csv_fpath)
        fnames = list(dataframe.iloc[:, 0])
        targets = list(dataframe.iloc[:, 1])

        return fnames, targets


if __name__ == '__main__':
    skin7 = Skin7(root="Skin7", train=True, fold_i=0)
    print(skin7.num_samples_per_cls)
