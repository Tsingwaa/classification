# import cv2
from os.path import join

import pandas as pd
import scipy.io as scio
import torch
import yaml
from data_loader.dataset.builder import DATASETS_ROOT, PROJECT_ROOT, Datasets
from PIL import Image
import random
import numpy as np

@Datasets.register_module("Flowers")
class Flowers(torch.utils.data.Dataset):
    num_classes = 102


    def __init__(self,
                 phase,
                 root="flowers",
                 transform=None,
                 imb_factor=0.1,
                 imb_type="exp",
                 seed=0,
                 **kwargs):
        # phase: "train" or "test"

        if "/" not in root:
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.transform = transform
        self.img_paths = []
        self.targets = []
        self.imb_factor = imb_factor
        self.imb_type = imb_type
        labels = scio.loadmat(join(root, 'imagelabels.mat'))
        labels_list = list(labels['labels'][0])
        l = {i:labels_list.count(i) for i in range( 1, max(labels_list)+1)}
        old_num = dict(sorted(l.items(),key=lambda d:d[1], reverse=True))
        old_reverse = list(old_num.keys())

        self.map_new2old = {}
        for i in range(len(old_reverse)):
            self.map_new2old[i] = old_reverse[i]

        self.map_old2new = {}
        for i in range(len(old_reverse)):
            self.map_old2new[old_reverse[i]] = i

        if self.phase == 'train':
            with open(join(root, 'train.txt'), 'r') as f:
                for line in f.readlines():
                    image_id, target = line[:11], int(line[12:])
                    self.img_paths.append(join(root, f'jpg/{image_id}.jpg'))
                    self.targets.append(self.map_old2new[target])
            self.old_num_samples_per_cls = [
                self.targets.count(i) for i in range(self.num_classes)
            ]
            np.random.seed(seed)
            self.num_samples_per_cls = self.get_img_num_per_cls()
            self.gen_imbalanced_data(self.num_samples_per_cls)
        else:
            with open(join(root, 'test.txt'), 'r') as f:
                for line in f.readlines():
                    image_id, target = line[:11], int(line[12:])
                    self.img_paths.append(join(root, f'jpg/{image_id}.jpg'))
                    # self.targets.append(target-1)
                    self.targets.append(self.map_old2new[target])
            self.num_samples_per_cls = [
                self.targets.count(i) for i in range(self.num_classes)
            ]
        
        self.group_mode = "class"

    def get_img_num_per_cls(self):
        max_num_samples = self.targets.count(0)
        # max_num_samples = int(len(self.img_paths) / self.num_classes)
        num_samples_per_cls = []

        if self.imb_type == 'exp':
            for class_index in range(self.num_classes):
                num_samples =\
                    max_num_samples * (self.imb_factor ** (
                        class_index / (self.num_classes - 1.0)))
                num_samples_per_cls.append(min(int(num_samples), self.old_num_samples_per_cls[class_index]))
        elif self.imb_type == 'step':
            half_num_classes = int(self.num_classes // 2)

            for class_index in range(self.num_classes):
                if class_index <= half_num_classes:
                    num_samples = max_num_samples
                else:
                    num_samples = int(max_num_samples * self.imb_factor)

                num_samples_per_cls.append(num_samples)
        else:
            # Original balance CIFAR dataset.
            num_samples_per_cls = [max_num_samples] * self.num_classes

        return num_samples_per_cls

    def gen_imbalanced_data(self, num_samples_per_cls):
        new_img_paths = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        self.img_paths = np.array(self.img_paths)
        classes = np.unique(targets_np)

        self.cls2nsamples = dict()

        for cls, n_samples in zip(classes, num_samples_per_cls):
            self.cls2nsamples[cls] = n_samples
            # random shuffle indexs and select num_samples of the class
            idx = np.where(targets_np == cls)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:n_samples]
            # generate new train pairs
            new_img_paths.extend(self.img_paths[selec_idx].tolist())
            new_targets.extend([
                cls,
            ] * n_samples)

        self.img_paths = new_img_paths
        self.targets = new_targets

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)



if __name__ == '__main__':
   print()