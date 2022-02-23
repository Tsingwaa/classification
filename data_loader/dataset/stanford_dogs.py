""" Stanford Dogs (Dog) Dataset
Created: Nov 15,2019 - Yuchong Gu
Revised: Nov 15,2019 - Yuchong Gu
"""
from os.path import join
from PIL import Image
from scipy.io import loadmat
from data_loader.dataset.builder import DATASETS_ROOT, PROJECT_ROOT, Datasets
import torch

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


if __name__ == '__main__':
    ds = StanfordDogs('train')
    # print(len(ds))

    for i in range(0, 1000):
        image, label = ds[i]
        # print(image.shape, label)
