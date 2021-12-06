# To ensure fairness, we use the same code in
# LDAM (https://github.com/kaidic/LDAM-DRW) &
# BBN (https://github.com/Megvii-Nanjing/BBN)
# to produce long-tailed CIFAR datasets.

import torchvision
from torchvision import transforms
import numpy as np
import PIL
# from pudb import set_trace
from data_loader.dataset.builder import Datasets


@Datasets.register_module("CIFAR10")
class CIFAR10_(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, data_root, train, transform=None, download=True,
                 **kwargs):
        super(CIFAR10_, self).__init__(
            root=data_root,
            train=train,
            transform=transform,
            download=download
        )


@Datasets.register_module("ImbalanceCIFAR10")
class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    def __init__(self, data_root, train, transform=None, download=True,
                 imb_type='exp', imb_factor=0.01, seed=0, **kwargs):
        super(ImbalanceCIFAR10, self).__init__(
            root=data_root,
            train=train,
            transform=transform,
            download=download
        )

        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.class_adapt = kwargs['class_adapt']
        self.seed = seed
        np.random.seed(self.seed)

        self.img_num_per_cls = self.get_img_num_per_cls()
        self.gen_imbalanced_data()

    def get_img_num_per_cls(self):
        img_max = len(self.data) / self.cls_num
        img_num_per_cls = []
        if self.imb_type == 'exp':
            for cls_idx in range(self.cls_num):
                num = img_max * (
                    self.imb_factor ** (cls_idx / (self.cls_num - 1.0))
                )
                img_num_per_cls.append(int(num))
        elif self.imb_type == 'step':
            # One step: the former half {img_max} imgs,
            # the latter half {img_max * imb_factor} imgs
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max * self.imb_factor))
        else:
            # Original balance CIFAR dataset.
            img_num_per_cls.extend([int(img_max)] * self.cls_num)

        return img_num_per_cls

    def gen_imbalanced_data(self):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.unique default output by increasing order. i.e. {class 0}: MAX.
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for cls_idx, num in zip(classes, self.img_num_per_cls):
            self.num_per_cls_dict[cls_idx] = num
            indexes = np.where(targets_np == cls_idx)[0]  # get index
            # Shuffle indexes for each class.
            np.random.shuffle(indexes)
            select_indexes = indexes[:num]
            new_data.append(self.data[select_indexes, ...])
            new_targets.extend([cls_idx, ] * num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = PIL.Image.fromarray(img)
        # set_trace()

        if self.transform is not None:
            percent = (1. + target) / 10. if self.class_adapt else None
            img = self.transform(img, percent=percent,
                                 mean=self.mean, std=self.std)
        return img, target

    @property
    def imgs_per_cls(self):
        return self.img_num_per_cls


@Datasets.register_module("ImbalanceCIFAR100")
class ImbalanceCIFAR100(ImbalanceCIFAR10):
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


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = ImbalanceCIFAR100(data_root='./data',
                                 train=True,
                                 download=True,
                                 transform=transform)
    # trainloader = iter(trainset)
    # data, label = next(trainloader)
    # import pdb
    # pdb.set_trace()
