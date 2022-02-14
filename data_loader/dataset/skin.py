# import cv2
from os.path import join

import pandas as pd
import torch
import yaml
from data_loader.dataset.builder import DATASETS_ROOT, PROJECT_ROOT, Datasets
from PIL import Image


@Datasets.register_module("Skin7")
class Skin7(torch.utils.data.Dataset):
    """Original image size: (3, 450, 600)"""
    num_classes = 7
    splitfold_mean_std = [
        
        ([0.5709, 0.5464, 0.7634], [0.1701, 0.1528, 0.1408]),
        ([0.5707, 0.5462, 0.7634], [0.1707, 0.1531, 0.1416]),
        ([0.5704, 0.5462, 0.7642], [0.1704, 0.1528, 0.1410]),
        ([0.5701, 0.5458, 0.7636], [0.1702, 0.1530, 0.1413]),
        ([0.5705, 0.5461, 0.7629], [0.1703, 0.1528, 0.1413]),
    ]

    def __init__(self,
                 phase,
                 root="Skin7",
                 fold_i=0,
                 transform=None,
                 **kwargs):
        # phase: "train" or "test"
        # fold_i: one of [0, 1, 2, 3, 4]

        if "/" not in root:
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.fold_i = fold_i
        self.transform = transform

        data_dir = join(root, 'ISIC2018_Task3_Training_Input')
        self.img_names, self.targets = self.get_data(fold_i, root, phase)

        self.img_paths = [
            join(data_dir, img_name) for img_name in self.img_names
        ]

        # 1: 1341 -> 0
        # 0: 223  -> 1
        # 4: 220  -> 2
        # 2: 103  -> 3
        # 3: 66   -> 4
        # 6: 29   -> 5
        # 5: 23   -> 6
        remap = {
            1: 0,
            0: 1,
            4: 2,
            2: 3,
            3: 4,
            6: 5,
            5: 6,
        }
        self.targets = [remap[target] for target in self.targets]
        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]
        # train:[5364, 890, 879, 411, 261, 113, 92]
        # test: [1341, 223, 220, 103, 66,  29,  23]
        self.group_mode = "class"

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            mean, std = self.splitfold_mean_std[self.fold_i]
            img = self.transform(img, mean=mean, std=std)

        return img, target

    def __len__(self):
        return len(self.targets)

    def get_data(self, fold_i, root, phase):
        """fetch the img names and targets"""

        fold_i += 1
        csv_fpath = join(root, "split_data", "origin_split_data",
                         f"split_data_{fold_i}_fold_{phase}.csv")
        dataframe = pd.read_csv(csv_fpath)
        img_names = list(dataframe.iloc[:, 0])
        targets = list(dataframe.iloc[:, 1])

        return img_names, targets


@Datasets.register_module("Skin8")
class Skin8(Skin7):
    num_classes = 8
    split_path = join(PROJECT_ROOT, "data_loader/data/Skin8/split_data.yaml")

    # [10300, 3617, 2658, 2099, 693, 502, 202, 191]
    splitfold_mean_std = [
        ([0.5243, 0.5298, 0.6669], [0.2321, 0.2237, 0.2458]),
        ([0.5253, 0.5304, 0.6690], [0.2319, 0.2231, 0.2456]),
        ([0.5240, 0.5294, 0.6674], [0.2324, 0.2239, 0.2463]),
        ([0.5244, 0.5298, 0.6676], [0.2324, 0.2238, 0.2459]),
        ([0.5242, 0.5296, 0.6684], [0.2320, 0.2234, 0.2461]),
    ]

    def __init__(self,
                 phase,
                 root="ISIC2019",
                 fold_i=0,
                 transform=None,
                 **kwargs):
        # phase: "train" or "test"
        # fold_i: one of [0, 1, 2, 3, 4]

        if "/" not in root:
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.fold_i = fold_i
        self.transform = transform

        data_dir = join(root, 'Data')
        self.img_names, self.targets = self.get_data(fold_i, phase)

        self.img_paths = [
            join(data_dir, img_name) for img_name in self.img_names
        ]

        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]
        self.group_mode = "class"

    def get_data(self, fold_i, phase):
        """fetch the img names and targets"""
        with open(self.split_path, "r") as f:
            fold2data = yaml.load(f, Loader=yaml.FullLoader)

        if phase == "train":
            data = []

            for fold_j in range(5):
                if fold_j != fold_i:
                    data.extend(fold2data[fold_j])
        else:  # "val" or "test"
            data = fold2data[fold_i]

        img_names = [img_name for img_name, target in data]
        targets = [target for img_name, target in data]

        return img_names, targets


if __name__ == '__main__':
    skin7 = Skin7(root="Skin7", train=True, fold_i=0)
    print(skin7.num_samples_per_cls)
