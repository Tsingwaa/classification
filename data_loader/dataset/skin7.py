# import os
from os.path import join

import pandas as pd
import torch
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from PIL import Image


def make_dataset(fold_i, data_root, phase):
    """fetch the img names and targets"""
    csv_fpath = join(
        data_root,
        "split_data/origin_split_data",
        f"split_data_{fold_i}_fold_{phase}.csv",
    )
    dataframe = pd.read_csv(csv_fpath)
    fnames = list(dataframe.iloc[:, 0])
    targets = list(dataframe.iloc[:, 1])

    return fnames, targets


@Datasets.register_module("Skin7")
class Skin7(torch.utils.data.Dataset):
    """Original image size: (3, 450, 600)"""
    num_classes = 7
    mean = [0.7626, 0.5453, 0.5714]
    std = [0.1404, 0.1519, 0.1685]

    def __init__(self, root, train, transform, fold_i, **kwargs):
        self.train = train
        self.phase = "train" if train else "test"
        self.transform = transform

        # Absolute path of Skin7 data root
        self.root = join(DATASETS_ROOT, root)
        # Image location
        self.data_dir = join(self.root, 'ISIC2018_Task3_Training_Input')

        self.img_fnames, self.targets = make_dataset(fold_i, self.root,
                                                     self.phase)

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

        print(f"===> Initialized fold-{fold_i} {self.phase}set...")

    def __getitem__(self, index):
        img_fname = self.img_fnames[index]
        target = self.targets[index]

        img_fpath = join(self.data_dir, img_fname)
        img = Image.open(img_fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, target

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    skin7 = Skin7(root="Skin7", train=True, transform=None, fold_i=1)
    print(skin7.num_samples_per_cls)
