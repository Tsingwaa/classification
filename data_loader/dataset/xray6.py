import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from PIL import Image


@Datasets.register_module("Xray6")
class Xray6(torch.utils.data.Dataset):
    num_classes = 6

    def __init__(self,
                 root,
                 train,
                 transform=None,
                 fold=0,
                 select_classes=[0, 1, 5, 7, 9, 11]):
        super(Xray6, self).__init__()

        self.root = os.path.join(DATASETS_ROOT, root)
        self.transform = transform
        self.train = train
        self.fold = fold
        self.select_classes = select_classes

        with open(self.root + '/split.json') as f:
            class_dict = json.load(f)

        self.fnames, self.labels = [], []

        if train:  # Train
            for i in range(5):
                if i == fold:
                    continue

                for d in class_dict[str(i)]:
                    if d[1] in select_classes:
                        self.fnames.append(d[0])
                        self.labels.append(d[1])
        else:  # Test
            for d in class_dict[str(fold)]:
                if d[1] in select_classes:
                    self.fnames.append(d[0])
                    self.labels.append(d[1])

        valid_labels = np.unique(self.labels)
        idx_to_class = {}

        for i in range(len(valid_labels)):
            idx_to_class[valid_labels[i]] = i

        for i in range(len(self.labels)):
            self.labels[i] = idx_to_class[self.labels[i]]
        self.num_per_cls = np.unique(self.labels, return_counts=True)[1]
        self.reciprocal_num_per_cls = 1.0 / self.num_per_cls
        self.class_balance_weight /= np.sum(self.reciprocal_num_per_cls)

    def __getitem__(self, index):
        img_fname, label = self.data[index], self.labels[index]
        img_fpath = os.path.join(self.root, img_fname)
        img = Image.open(img_fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)


def get_xray14_dataset(data_root='Xray14/categories',
                       bs=64,
                       train=True,
                       fold=0,
                       select_classes=[0, 1, 5, 7, 9, 11]):
    # mean = (0.527, 0.527, 0.527)
    # std = (0.203, 0.203, 0.203)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if train:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            # transforms.RandomCrop(224),
            # transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            # transforms.RandomRotation([-180, 180]),
            # transforms.RandomAffine(
            # [-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        transform = T.Compose([
            T.Resize((224, 224)),
            #    transforms.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    dataset = Xray6(data_root,
                    train,
                    transform,
                    fold,
                    select_classes=select_classes)
    weights = dataset.get_num_per_cls()
    weights = torch.tensor(weights, dtype=torch.float)
    # weights = 1.0 / weights
    # weights /= weights.sum()
    # dl = torch.utils.data.DataLoader(dataset, bs, train,
    # num_workers=16, pin_memory=True, drop_last=train)

    return dataset, weights
