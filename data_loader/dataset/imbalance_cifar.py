# To ensure fairness, we use the same code in
# LDAM (https://github.com/kaidic/LDAM-DRW) &
# BBN (https://github.com/Megvii-Nanjing/BBN)
# to produce long-tailed CIFAR datasets.

import torchvision
import torchvision.transforms as transforms
import numpy as np
from data_loader.dataset.builder import Datasets


@Datasets.register_module("ImbalanceCifar10")
class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    def __init__(self,
                 data_root,
                 imb_type='exp',
                 imb_factor=0.01,
                 rand_seed=0,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        super(IMBALANCECIFAR10, self).__init__(data_root,
                                               train,
                                               transform,
                                               target_transform,
                                               download)
        np.random.seed(rand_seed)
        img_num_list = self.get_img_num_per_cls(self.cls_num,
                                                imb_type,
                                                imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            # One step: the former half {img_max} imgs,
            # the latter half {img_max * imb_factor} imgs
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            # Original balance CIFAR dataset.
            img_num_per_cls.extend([int(img_max)] * cls_num)

        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.unique default output by increasing order. i.e. {class 0}: MAX.
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for cls, num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[cls] = num
            indexes = np.where(targets_np == cls)[0]  # get index
            # Shuffle indexes for each class.
            np.random.shuffle(indexes)
            select_indexes = indexes[:num]
            new_data.append(self.data[select_indexes])
            new_targets.extend([cls, ] * num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


@Datasets.register_module("ImbalanceCifar100")
class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = IMBALANCECIFAR100(data_root='./data', train=True,
                                 download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb
    pdb.set_trace()
