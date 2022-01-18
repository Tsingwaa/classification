import os

import numpy as np
import torch
import torch.utils.data as data
from data_loader.dataset.builder import Datasets
from PIL import Image
from torchvision import transforms as T


@Datasets.register_module("MedMNIST")
class MedMNIST(data.Dataset):
    cls_num = 0

    def __init__(self,
                 root,
                 sub,
                 split='train',
                 transform=None,
                 imb_factor=0.1,
                 **kwargs):
        super(MedMNIST, self).__init__()
        self.transform = transform if transform is not None else lambda x: x
        self.data = np.load(os.path.join(
            root, '{}.npz'.format(sub)))[split + '_images'].squeeze().astype(
                np.uint8)
        self.labels = np.load(os.path.join(
            root,
            '{}.npz'.format(sub)))[split + '_labels'].squeeze().astype(np.long)
        self.sub = sub

        if split == 'train':
            if imb_factor < 1:
                print("create_imbalance")
                self.create_imbalance(imb_factor)
        else:
            pass
            # self.create_balance()

        if sub == 'pathmnist':
            self.cls_num = 9
        elif sub == 'octmnist':
            self.cls_num = 4
        elif sub == 'organmnist_sagittal':
            self.cls_num = 11

    def create_imbalance(self, imb_factor):
        tmp_data, tmp_label = [], []
        classes, class_count = np.unique(self.labels, return_counts=True)
        sort_id = np.argsort(class_count)[::-1]
        classes, class_count = classes[sort_id], class_count[sort_id]
        max_num = class_count[0]
        # self.cls_num = classes.size

        for i in range(len(classes)):
            imgs = self.data[self.labels == classes[i]]
            labels = self.labels[self.labels == classes[i]]
            num = int(max_num * (imb_factor**(i / (len(classes) - 1.0))))
            idx = np.arange(len(labels))

            if num < class_count[i]:
                np.random.shuffle(idx)
                idx = idx[:num]
            tmp_data.append(imgs[idx])
            tmp_label.append(labels[idx])

        self.data = np.concatenate(tmp_data)
        self.labels = np.concatenate(tmp_label)

    def create_balance(self):
        tmp_data, tmp_label = [], []
        classes, class_count = np.unique(self.labels, return_counts=True)
        min_num = class_count.min()
        # self.cls_num = classes.size

        for i in range(len(classes)):
            imgs = self.data[self.labels == classes[i]]
            labels = self.labels[self.labels == classes[i]]
            tmp_data.append(imgs[:min_num])
            tmp_label.append(labels[:min_num])

        self.data = np.concatenate(tmp_data)
        self.labels = np.concatenate(tmp_label)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        # print(img)

        if self.sub in ['octmnist', 'organmnist_sagittal']:
            img = img.convert('RGB')
        img = self.transform(img)
        label = self.labels[index]

        return img, label

    def __len__(self):
        return self.labels.shape[0]


def get_medmnist_dataset(split='train', imb_factor=0.1, sub='pathmnist'):
    root = '/home/waa/Disk/Warehouse/Datasets/medmnist'

    if split == 'train':
        transform = T.Compose([
            T.Resize(32),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    else:
        transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
        ])
    # train_flag = (split == 'train')
    ds = MedMNIST(root, sub, split, transform, imb_factor)
    # dl = data.DataLoader(ds,
    #                      bs,
    #                      train_flag,
    #                      num_workers=2,
    #                      drop_last=train_flag,
    #                      pin_memory=True)

    weights = np.unique(ds.labels, return_counts=True)[1]
    weights = torch.from_numpy(weights).to(torch.float)

    return ds, weights
