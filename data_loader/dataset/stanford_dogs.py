""" Stanford Dogs (Dog) Dataset
Created: Nov 15,2019 - Yuchong Gu
Revised: Nov 15,2019 - Yuchong Gu
"""
from os.path import join
from PIL import Image
from scipy.io import loadmat
from data_loader.dataset.builder import DATASETS_ROOT, PROJECT_ROOT, Datasets
import torch
import numpy as np
@Datasets.register_module("StanfordDogs")
class StanfordDogs(torch.utils.data.Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Dogs images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', root='dogs', transform=None, **kwargs):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.num_classes = 120
        self.transform = transform
        if "/" not in root:
            root = join(DATASETS_ROOT, root)
        self.root = root

        if phase == 'train':
            list_path = join(root, 'train_list.mat')
        else:
            list_path = join(root, 'test_list.mat')

        list_mat = loadmat(list_path)
        self.img_paths = [f.item().item() for f in list_mat['file_list']]
        self.targets = [f.item() for f in list_mat['labels']]

        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(1, self.num_classes+1)
        ]
        
        self.group_mode = "class"

    def __getitem__(self, item):
        # image
        image = Image.open(join(self.root,
                                        'Images', self.img_paths[item])).convert(
                                            'RGB')  # (C, H, W)
        if self.transform is not None:
            image = self.transform(image)

        # return image and label

        return image, self.targets[item] - 1  # count begin from zero

    def __len__(self):
        return len(self.img_paths)

@Datasets.register_module("Imbalance_StanfordDogs")
class Imbalance_StanfordDogs(torch.utils.data.Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Dogs images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', imb_type='exp',
                 imb_factor=0.1, root='dogs', seed=0, transform=None, **kwargs):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.num_classes = 120
        self.transform = transform
        if "/" not in root:
            root = join(DATASETS_ROOT, root)
        self.root = root
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        if phase == 'train':
            list_path = join(root, 'train_list.mat')
        else:
            list_path = join(root, 'test_list.mat')

        list_mat = loadmat(list_path)
        self.img_paths = [f.item().item() for f in list_mat['file_list']]
        self.targets = [f.item()-1 for f in list_mat['labels']]
        if phase == "train":
            self.old_num_samples_per_cls = [
                self.targets.count(i) for i in range(self.num_classes)
            ]
            np.random.seed(seed)
            self.num_samples_per_cls = self.get_img_num_per_cls()
            self.gen_imbalanced_data(self.num_samples_per_cls)
        else:
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
    def __getitem__(self, item):
        # image
        image = Image.open(join(self.root,
                                        'Images', self.img_paths[item])).convert(
                                            'RGB')  # (C, H, W)
        if self.transform is not None:
            image = self.transform(image)

        # return image and label

        return image, self.targets[item]  # count begin from zero

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    ds = StanfordDogs('train')
    # print(len(ds))

    for i in range(0, 1000):
        image, label = ds[i]
        # print(image.shape, label)
