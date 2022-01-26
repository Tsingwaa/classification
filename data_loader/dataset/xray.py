import json
from os.path import join

# import cv2
import torch
# import torchvision.transforms as T
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from PIL import Image


@Datasets.register_module("Xray14")
class Xray14(torch.utils.data.Dataset):
    num_classes = 14

    splitfold_mean_std = [
        ([0.4924, 0.4924, 0.4924], [0.2488, 0.2488, 0.2488]),
        ([0.4924, 0.4924, 0.4924], [0.2485, 0.2485, 0.2485]),
        ([0.4924, 0.4924, 0.4924], [0.2487, 0.2487, 0.2487]),
        ([0.4929, 0.4929, 0.4929], [0.2487, 0.2487, 0.2487]),
        ([0.4925, 0.4925, 0.4925], [0.2490, 0.2490, 0.2490]),
    ]

    def __init__(self,
                 root,
                 phase,
                 fold_i=0,
                 transform=None,
                 select_classes=list(range(14))):
        super(Xray14, self).__init__()

        if "/" not in root:  # 给定root为数据集根目录
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.fold_i = fold_i
        self.transform = transform

        splitfold_path = join(root, "categories/split.json")
        self.img_names, self.targets = self.get_data(splitfold_path, fold_i,
                                                     phase, select_classes)

        data_dir = join(root, "categories")
        self.img_paths = [
            join(data_dir, img_name) for img_name in self.img_names
        ]

        # 2:  9550->0
        # 0:  4210->1
        # 11: 3955->2
        # 10: 2705->3
        # 8:  2195->4
        # 9:  2135->5
        # 3:  1310->6
        # 13: 1125->7
        # 1:  1090->8
        # 5:  895 ->9
        # 4:  725 ->10
        # 12: 630 ->11
        # 6:  305 ->12
        # 7:  110 ->13
        remap = {
            2: 0,
            0: 1,
            11: 2,
            10: 3,
            8: 4,
            9: 5,
            3: 6,
            13: 7,
            1: 8,
            5: 9,
            4: 10,
            12: 11,
            6: 12,
            7: 13,
        }

        self.targets = [remap[target] for target in self.targets]
        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]
        self.group_mode = "class"

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            mean, std = self.splitfold_mean_std[self.fold_i]
            img = self.transform(img, mean=mean, std=std)

        return img, target

    def __len__(self):
        return len(self.targets)

    def get_data(self, splitfold_path, fold_i, phase, select_classes):

        with open(splitfold_path, "r") as f:
            fold2data = json.load(f)
            # {"0":[['Atelectasis/00003548_003.png', 0],...,], "1":...}

        if phase == 'train':
            data = []

            for fold_j in range(5):
                if fold_j != fold_i:
                    data.extend(fold2data[str(fold_j)])
        elif phase in ['val', 'test']:
            data = fold2data[str(fold_i)]
        else:
            raise TypeError

        # fnames(str list): "class/fname"
        # targets(int list): elem in select_classes
        img_names = [
            img_name for img_name, target in data if target in select_classes
        ]
        targets = [
            target for img_name, target in data if target in select_classes
        ]

        return img_names, targets


@Datasets.register_module("Xray6")
class Xray6(torch.utils.data.Dataset):
    num_classes = 6
    splitfold_mean_std = [([0.5028, 0.5028, 0.5028], [0.2496, 0.2496, 0.2496]),
                          ([0.5032, 0.5032, 0.5032], [0.2500, 0.2500, 0.2500]),
                          ([0.5029, 0.5029, 0.5029], [0.2499, 0.2499, 0.2499]),
                          ([0.5031, 0.5031, 0.5031], [0.2504, 0.2504, 0.2504]),
                          ([0.5031, 0.5031, 0.5031], [0.2500, 0.2500, 0.2500])]

    def __init__(self,
                 root,
                 phase,
                 fold_i=0,
                 transform=None,
                 select_classes=[0, 1, 5, 7, 9, 11]):
        super(Xray6, self).__init__()

        if "/" not in root:  # 给定root为数据集根目录
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.fold_i = fold_i
        self.transform = transform

        splitfold_path = join(root, "categories/split.json")
        self.img_names, self.targets = self.get_data(splitfold_path, fold_i,
                                                     phase, select_classes)

        data_dir = join(root, "categories")
        self.img_paths = [
            join(data_dir, img_name) for img_name in self.img_names
        ]

        # 0: 4210 -> 0
        # 11:3955 -> 1
        # 9: 2135 -> 2
        # 1: 1090 -> 3
        # 5: 895  -> 4
        # 7: 110  -> 5
        remap = {
            0: 0,
            11: 1,
            9: 2,
            1: 3,
            5: 4,
            7: 5,
        }

        self.targets = [remap[target] for target in self.targets]
        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]
        self.group_mode = "class"

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            mean, std = self.splitfold_mean_std[self.fold_i]
            img = self.transform(img, mean=mean, std=std)

        return img, target

    def __len__(self):
        return len(self.targets)

    def get_data(self, splitfold_path, fold_i, phase, select_classes):

        with open(splitfold_path, "r") as f:
            fold2data = json.load(f)
            # {"0":[['Atelectasis/00003548_003.png', 0],...,], "1":...}

        if phase == 'train':
            data = []

            for fold_j in range(5):
                if fold_j != fold_i:
                    data.extend(fold2data[str(fold_j)])
        elif phase == 'val':
            data = fold2data[str(fold_i)]
        else:
            raise NotImplementedError

        # fnames(str list): "class/fname"
        # targets(int list): elem in select_classes
        img_names = [
            img_name for img_name, target in data if target in select_classes
        ]
        targets = [
            target for img_name, target in data if target in select_classes
        ]

        return img_names, targets


if __name__ == "__main__":
    xray6 = Xray6(data_root="Xray14", )
