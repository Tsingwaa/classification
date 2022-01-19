import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from PIL import Image


@Datasets.register_module("SD198")
class SD198(torch.utils.data.Dataset):
    num_classes = 198

    # dataset_name = "SD198"

    def __init__(self, root='SD198', train=True, transform=None, fold=0):

        root = os.path.join(DATASETS_ROOT, root)
        self.train = train
        self.data_dir = os.path.join(root, 'images')
        self.data, self.targets = self.get_data(fold, root)
        class_idx_path = os.path.join(root, 'class_idx.npy')
        self.classes = self.get_classes_name(class_idx_path)
        self.classes = [class_name for class_name, _ in self.classes]

        if self.train:
            self.train_labels = torch.LongTensor(self.targets)
            self.img_num = [
                self.targets.count(i) for i in range(self.num_classes)
            ]
        else:
            self.test_labels = torch.LongTensor(self.targets)

        Resize_img = 300
        Crop_img = 224
        self.labels = self.targets
        mean = [0.5896185, 0.4765919, 0.45172438]
        std = [0.26626918, 0.24757613, 0.24818243]

        transform_train = T.Compose([
            T.Resize(Resize_img),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            # transforms.RandomRotation([-180, 180]),
            # T.RandomAffine([-180, 180], translate=[0.1, 0.1],
            #                scale=[0.7, 1.3]),
            T.RandomCrop(Crop_img),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        transform_test = T.Compose([
            T.Resize(Resize_img),
            T.CenterCrop(Crop_img),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transform_train if self.train else transform_test

    def __getitem__(self, index):
        img_fname = self.data[index]
        img_fpath = os.path.join(self.data_dir, img_fname)
        target = self.targets[index]
        img = Image.open(img_fpath).convert('RGB')
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data(self, fold, data_dir):
        if self.train:
            fname = '8_2_split/train_{}.txt'.format(fold)
        else:
            fname = '8_2_split/val_{}.txt'.format(fold)

        img_lst_fpath = os.path.join(data_dir, fname)
        df = pd.read_csv(img_lst_fpath, sep=" ")
        raw_data = df.values

        fnames = []
        labels = []

        for fname, label in raw_data:
            fnames.append(fname)
            labels.append(label)

        return fnames, labels

    @staticmethod
    def get_classes_name(data_dir):
        classes_name = np.load(data_dir)

        return classes_name


if __name__ == '__main__':
    mean = (0.592, 0.479, 0.451)
    std = (0.265, 0.245, 0.247)
    transform = T.Compose(
        [T.Resize((224, 224)),
         T.ToTensor(),
         T.Normalize(mean=mean, std=std)])
    trainset = SD198(train=True, transform=transform, fold=1)

    loader = torch.utils.data.DataLoader(trainset,
                                         batch_size=16,
                                         shuffle=True,
                                         num_workers=8)

    for data in loader:
        images, labels = data
        print('images:', images.size())
        print('labels', labels.size())
