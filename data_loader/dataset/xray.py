import random
from os.path import join

import torch
import yaml
from data_loader.dataset.builder import DATASETS_ROOT, PROJECT_ROOT, Datasets
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
    select_classes = list(range(14))
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

    def __init__(self, phase, root="Xray14", fold_i=0, transform=None):
        # phase: "train" or "test"
        super(Xray14, self).__init__()

        if "/" not in root:  # 给定root为数据集根目录
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.fold_i = fold_i
        self.transform = transform

        splitfold_path = join(PROJECT_ROOT,
                              "data_loader/data/Xray/split_data.yaml")
        self.img_names, self.targets = self.get_data(
            splitfold_path,
            fold_i,
            phase,
            self.select_classes,
        )

        data_dir = join(root, "categories")
        self.img_paths = [
            join(data_dir, img_name) for img_name in self.img_names
        ]

        self.targets = [self.remap[target] for target in self.targets]
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
            fold2data = yaml.load(f, Loader=yaml.FullLoader)
            # {"0":[['Atelectasis/00003548_003.png', 0],...,], "1":...}

        if phase == 'train':
            data = []

            for fold_j in range(5):
                if fold_j != fold_i:
                    data.extend(fold2data[fold_j])
        elif phase in ['val', 'test']:
            data = fold2data[fold_i]
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


@Datasets.register_module("Xray13")
class Xray13(Xray14):
    select_classes = [i for i in range(14) if i != 2]
    num_classes = 13
    splitfold_mean_std = [
        ([0.4895, 0.4895, 0.4895], [0.2492, 0.2492, 0.2492]),
        ([0.4896, 0.4896, 0.4896], [0.2489, 0.2489, 0.2489]),
        ([0.4898, 0.4898, 0.4898], [0.2491, 0.2491, 0.2491]),
        ([0.4903, 0.4903, 0.4903], [0.2491, 0.2491, 0.2491]),
        ([0.4898, 0.4898, 0.4898], [0.2493, 0.2493, 0.2493]),
    ]

    # 2:  9550  excluded

    # 0:  4210->0
    # 11: 3955->1
    # 10: 2705->2
    # 8:  2195->3
    # 9:  2135->4
    # 3:  1310->5
    # 13: 1125->6
    # 1:  1090->7
    # 5:  895 ->8
    # 4:  725 ->9
    # 12: 630 ->10
    # 6:  305 ->11
    # 7:  110 ->12
    remap = {
        0: 0,
        11: 1,
        10: 2,
        8: 3,
        9: 4,
        3: 5,
        13: 6,
        1: 7,
        5: 8,
        4: 9,
        12: 10,
        6: 11,
        7: 12,
    }


@Datasets.register_module("Xray9")
class Xray9(Xray14):

    select_classes = [0, 1, 5, 7, 8, 9, 10, 11, 12]
    num_classes = 9
    splitfold_mean_std = [
        ([0.4895, 0.4895, 0.4895], [0.2492, 0.2492, 0.2492]),
        ([0.4896, 0.4896, 0.4896], [0.2489, 0.2489, 0.2489]),
        ([0.4898, 0.4898, 0.4898], [0.2491, 0.2491, 0.2491]),
        ([0.4903, 0.4903, 0.4903], [0.2491, 0.2491, 0.2491]),
        ([0.4898, 0.4898, 0.4898], [0.2493, 0.2493, 0.2493]),
    ]

    # 0,  Atelectasis 肺不张
    # 11, Effusion 胸腔积液
    # 10, Nodule  肺结节
    # 8,  Pneumothorax 气胸
    # 9,  Mass 肿块
    # 1,  Cardiomegaly 心脏肥大
    # 5,  Emphysema 肺气肿
    # 12, Edema 肺水肿
    # 7,  Hernia 肺疝

    # 0:  4210->0
    # 11: 3955->1
    # 10: 2705->2
    # 8:  2195->3
    # 9:  2135->4
    # 1:  1090->5
    # 5:  895 ->6
    # 12: 630 ->7
    # 7:  110 ->8
    remap = {
        0: 0,
        11: 1,
        10: 2,
        8: 3,
        9: 4,
        1: 5,
        5: 6,
        12: 7,
        7: 8,
    }


@Datasets.register_module("Xray14_BBN")
class Xray14_BBN(torch.utils.data.Dataset):
    num_classes = 14

    splitfold_mean_std = [
        ([0.4924, 0.4924, 0.4924], [0.2488, 0.2488, 0.2488]),
        ([0.4924, 0.4924, 0.4924], [0.2485, 0.2485, 0.2485]),
        ([0.4924, 0.4924, 0.4924], [0.2487, 0.2487, 0.2487]),
        ([0.4929, 0.4929, 0.4929], [0.2487, 0.2487, 0.2487]),
        ([0.4925, 0.4925, 0.4925], [0.2490, 0.2490, 0.2490]),
    ]
    select_classes = list(range(14))
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

    def __init__(self,
                 phase,
                 dual_sample=False,
                 dual_sample_type=None,
                 root="Xray14",
                 fold_i=0,
                 transform=None):
        # phase: "train" or "test"
        super(Xray14_BBN, self).__init__()

        if "/" not in root:  # 给定root为数据集根目录
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.fold_i = fold_i
        self.transform = transform
        self.dual_sample = True if dual_sample and phase == 'train' else False
        self.dual_sample_type = dual_sample_type
        splitfold_path = join(PROJECT_ROOT,
                              "data_loader/data/Xray/split_data.yaml")
        self.img_names, self.targets = self.get_data(
            splitfold_path,
            fold_i,
            phase,
            self.select_classes,
        )

        data_dir = join(root, "categories")
        self.img_paths = [
            join(data_dir, img_name) for img_name in self.img_names
        ]

        self.targets = [self.remap[target] for target in self.targets]
        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]
        self.group_mode = "class"
        if self.dual_sample:
            self.class_weight, self.sum_weight = self.get_weight(
                self.get_annotations(), self.num_classes)
            self.class_dict = self._get_class_dict()

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')

        meta = dict()

        if self.dual_sample:
            if self.dual_sample_type == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.dual_sample_type == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.dual_sample_type == "uniform":
                sample_index = random.randint(0, self.__len__() - 1)

            sample_img, sample_label = self.img_paths[
                sample_index], self.targets[sample_index]
            sample_img = Image.open(sample_img).convert('RGB')
            sample_img = self.transform(sample_img)

            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        if self.transform is not None:
            mean, std = self.splitfold_mean_std[self.fold_i]
            img = self.transform(img, mean=mean, std=std)

        return img, target, meta

    def __len__(self):
        return len(self.targets)

    def get_data(self, splitfold_path, fold_i, phase, select_classes):

        with open(splitfold_path, "r") as f:
            fold2data = yaml.load(f, Loader=yaml.FullLoader)
            # {"0":[['Atelectasis/00003548_003.png', 0],...,], "1":...}

        if phase == 'train':
            data = []

            for fold_j in range(5):
                if fold_j != fold_i:
                    data.extend(fold2data[fold_j])
        elif phase in ['val', 'test']:
            data = fold2data[fold_i]
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

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if cat_id not in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i


@Datasets.register_module("Xray9_BBN")
class Xray9_BBN(Xray14_BBN):

    select_classes = [0, 1, 5, 7, 8, 9, 10, 11, 12]
    num_classes = 9
    splitfold_mean_std = [
        ([0.4895, 0.4895, 0.4895], [0.2492, 0.2492, 0.2492]),
        ([0.4896, 0.4896, 0.4896], [0.2489, 0.2489, 0.2489]),
        ([0.4898, 0.4898, 0.4898], [0.2491, 0.2491, 0.2491]),
        ([0.4903, 0.4903, 0.4903], [0.2491, 0.2491, 0.2491]),
        ([0.4898, 0.4898, 0.4898], [0.2493, 0.2493, 0.2493]),
    ]

    # 0:  4210->0
    # 11: 3955->1
    # 10: 2705->2
    # 8:  2195->3
    # 9:  2135->4
    # 1:  1090->5
    # 5:  895 ->6
    # 12: 630 ->7
    # 7:  110 ->8
    remap = {
        0: 0,
        11: 1,
        10: 2,
        8: 3,
        9: 4,
        1: 5,
        5: 6,
        12: 7,
        7: 8,
    }


@Datasets.register_module("Xray8")
class Xray8(Xray14):
    select_classes = [0, 1, 5, 7, 8, 9, 10, 11]
    num_classes = 8
    splitfold_mean_std = [
        ([0.5011, 0.5011, 0.5011], [0.2503, 0.2503, 0.2503]),
        ([0.5007, 0.5007, 0.5007], [0.2500, 0.2500, 0.2500]),
        ([0.5011, 0.5011, 0.5011], [0.2503, 0.2503, 0.2503]),
        ([0.5010, 0.5010, 0.5010], [0.2502, 0.2502, 0.2502]),
        ([0.5012, 0.5012, 0.5012], [0.2506, 0.2506, 0.2506]),
    ]

    # 0: 4210 -> 0
    # 11:3955 -> 1
    # 10:2705 -> 2
    # 8: 2195 -> 3
    # 9: 2135 -> 4
    # 1: 1090 -> 5
    # 5: 895  -> 6
    # 7: 110  -> 7
    remap = {
        0: 0,
        11: 1,
        10: 2,
        8: 3,
        9: 4,
        1: 5,
        5: 6,
        7: 7,
    }


@Datasets.register_module("Xray6")
class Xray6(Xray14):
    select_classes = [0, 1, 5, 7, 9, 11]
    num_classes = 6
    splitfold_mean_std = [
        ([0.5028, 0.5028, 0.5028], [0.2496, 0.2496, 0.2496]),
        ([0.5032, 0.5032, 0.5032], [0.2500, 0.2500, 0.2500]),
        ([0.5029, 0.5029, 0.5029], [0.2499, 0.2499, 0.2499]),
        ([0.5031, 0.5031, 0.5031], [0.2504, 0.2504, 0.2504]),
        ([0.5031, 0.5031, 0.5031], [0.2500, 0.2500, 0.2500]),
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


if __name__ == "__main__":
    xray6 = Xray6(data_root="Xray14", )
