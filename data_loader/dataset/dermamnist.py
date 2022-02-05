from os.path import join

from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from PIL import Image

from med_mnist import DermaMNIST


@Datasets.register_module("DermaMNIST")
class MyDermaMNIST(DermaMNIST):
    num_classes = 7

    def __init__(self, root, phase, fold_i=0, transform=None, **kwargs):
        if "/" not in root:
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.transform = transform

        super(DermaMNIST, self).__init__(
            root=root,
            split=phase,
        )

        self.data = self.imgs

        # 5: 4693 -> 0
        # 4: 779  -> 1
        # 2: 769  -> 2
        # 1: 359  -> 3
        # 0: 228  -> 4
        # 6: 99   -> 5
        # 3: 80   -> 6
        remap = {
            5: 0,
            4: 1,
            2: 2,
            1: 3,
            0: 4,
            6: 5,
            3: 6,
        }

        self.targets = [remap[target] for target in self.labels]

        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]

        self.group_mode = "class"

    def __get_item__(self, index):
        img, target = self.imgs[index], self.targets[index]
        img = Image.fromarray(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img, mean=self.mean, std=self.std)

        return img, target

    def __len__(self):
        return len(self.targets)
