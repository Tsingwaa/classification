from os.path import join

import numpy as np
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from PIL import Image
from torch.utils.data import Dataset


# Dataset
@Datasets.register_module("ImageNet_LT")
class LT_Dataset(Dataset):
    num_classes = 1000
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self,
                 root,
                 img_lst_path,
                 phase='train',
                 transform=None,
                 imb_type='exp',
                 map_fpath=''):
        self.img_paths = []
        self.targets = []
        self.transform = transform
        self.map = np.load(map_fpath)

        if "/" not in root:
            root = join(DATASETS_ROOT, root)

        with open(img_lst_path) as f:
            for line in f:
                self.img_paths.append(join(root, line.split()[0]))
                self.targets.append(self.map[int(line.split()[1])])

        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]
        self.class_weight = self.get_class_weight()
        self.indexes_per_cls = self.get_indexes_per_cls()
        self.group_mode = "shot"

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        target = self.targets[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_class_list(self):
        return self.class_list

    def get_class_weight(self):
        num_samples_per_cls = np.array(self.num_samples_per_cls)
        num_samples = np.sum(num_samples_per_cls)
        weight = num_samples / (self.num_classes * num_samples_per_cls)
        weight /= np.sum(weight)

        return weight

    def get_indexes_per_cls(self):
        indexes_per_cls = []

        for i in range(self.num_classes):
            indexes = np.where(np.array(self.targets) == i)[0].tolist()
            indexes_per_cls.append(indexes)

        return indexes_per_cls
