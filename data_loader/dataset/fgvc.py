# import cv2
import random
from os.path import join

import pandas as pd
import torch
# import yaml
from data_loader.dataset.builder import DATASETS_ROOT, Datasets
from PIL import Image

# ('Boeing', 1466),
# ('Airbus', 867),
# ('McDonnell Douglas', 467),
# ('Embraer', 467),
# ('de Havilland', 334),
# ('British Aerospace', 267),
# ('Douglas Aircraft Company', 267),
# ('Canadair', 267),
# ('Cessna', 266),
# ('Fokker', 200),
# ('Beechcraft', 134),
# ('Lockheed Corporation', 134),
# ('Saab', 134),
# ('ATR', 133),
# ('Dassault Aviation', 133),
# ('Gulfstream Aerospace', 133),
# ('Tupolev', 133),
# ('Antonov', 67),
# ('Dornier', 67),
# ('Lockheed Martin', 67),
# ('Bombardier Aerospace', 67),
# ('Piper', 67),
# ('Panavia', 67),
# ('Yakovlev', 67),
# ('Robin', 66),
# ('Eurofighter', 66),
# ('Ilyushin', 66),
# ('Fairchild', 66),
# ('Cirrus Aircraft', 66),
# ('Supermarine', 66)]


@Datasets.register_module("FGVC")
class FGVC(torch.utils.data.Dataset):
    """Original image size: (3, 450, 600)"""
    num_classes = 30

    name2label = {
        'Boeing': 0,
        'Airbus': 1,
        'McDonnell Douglas': 2,
        'Embraer': 3,
        'de Havilland': 4,
        'British Aerospace': 5,
        'Douglas Aircraft Company': 6,
        'Canadair': 7,
        'Cessna': 8,
        'Fokker': 9,
        'Beechcraft': 10,
        'Lockheed Corporation': 11,
        'Saab': 12,
        'ATR': 13,
        'Dassault Aviation': 14,
        'Gulfstream Aerospace': 15,
        'Tupolev': 16,
        'Antonov': 17,
        'Dornier': 18,
        'Lockheed Martin': 19,
        'Bombardier Aerospace': 20,
        'Piper': 21,
        'Panavia': 22,
        'Yakovlev': 23,
        'Robin': 24,
        'Eurofighter': 25,
        'Ilyushin': 26,
        'Fairchild': 27,
        'Cirrus Aircraft': 28,
        'Supermarine': 29
    }

    label2name = {
        0: 'Boeing',
        1: 'Airbus',
        2: 'McDonnell Douglas',
        3: 'Embraer',
        4: 'de Havilland',
        5: 'British Aerospace',
        6: 'Douglas Aircraft Company',
        7: 'Canadair',
        8: 'Cessna',
        9: 'Fokker',
        10: 'Beechcraft',
        11: 'Lockheed Corporation',
        12: 'Saab',
        13: 'ATR',
        14: 'Dassault Aviation',
        15: 'Gulfstream Aerospace',
        16: 'Tupolev',
        17: 'Antonov',
        18: 'Dornier',
        19: 'Lockheed Martin',
        20: 'Bombardier Aerospace',
        21: 'Piper',
        22: 'Panavia',
        23: 'Yakovlev',
        24: 'Robin',
        25: 'Eurofighter',
        26: 'Ilyushin',
        27: 'Fairchild',
        28: 'Cirrus Aircraft',
        29: 'Supermarine'
    }

    def __init__(self,
                 phase,
                 root="fgvc-aircraft-2013b",
                 transform=None,
                 **kwargs):
        # phase: "train" or "test"

        if "/" not in root:
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.transform = transform
        self.img_paths = []
        self.targets = []
        if self.phase == 'train':
            with open(join(root, 'data/images_manufacturer_trainval.txt'),
                      'r') as f:
                for line in f.readlines():
                    image_id, target = line[:7], line[8:-1]
                    self.img_paths.append(
                        join(root, f'data/images/{image_id}.jpg'))
                    self.targets.append(self.name2label[target])
        else:
            with open(join(root, 'data/images_manufacturer_test.txt'),
                      'r') as f:
                for line in f.readlines():
                    image_id, target = line[:7], line[8:-1]
                    self.img_paths.append(
                        join(root, f'data/images/{image_id}.jpg'))
                    self.targets.append(self.name2label[target])

        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]

        self.group_mode = "class"

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)


# @Datasets.register_module("FGVC_BBN")
# class FGVC_BBN(torch.utils.data.Dataset):
#     """Original image size: (3, 450, 600)"""
#     num_classes = 7
#     splitfold_mean_std = [
#         ([0.5709, 0.5464, 0.7634], [0.1701, 0.1528, 0.1408]),
#         ([0.5707, 0.5462, 0.7634], [0.1707, 0.1531, 0.1416]),
#         ([0.5704, 0.5462, 0.7642], [0.1704, 0.1528, 0.1410]),
#         ([0.5701, 0.5458, 0.7636], [0.1702, 0.1530, 0.1413]),
#         ([0.5705, 0.5461, 0.7629], [0.1703, 0.1528, 0.1413]),
#     ]

#     def __init__(self,
#                  phase,
#                  dual_sample=False,
#                  dual_sample_type=None,
#                  root="Skin7",
#                  fold_i=0,
#                  transform=None,
#                  **kwargs):
#         # phase: "train" or "test"
#         # fold_i: one of [0, 1, 2, 3, 4]

#         if "/" not in root:
#             root = join(DATASETS_ROOT, root)

#         self.phase = phase
#         self.fold_i = fold_i
#         self.transform = transform

#         data_dir = join(root, 'ISIC2018_Task3_Training_Input')
#         self.img_names, self.targets = self.get_data(fold_i, root, phase)
#         self.dual_sample = True if dual_sample and phase == 'train' else False
#         self.dual_sample_type = dual_sample_type
#         self.img_paths = [
#             join(data_dir, img_name) for img_name in self.img_names
#         ]

#         # 1: 1341 -> 0
#         # 0: 223  -> 1
#         # 4: 220  -> 2
#         # 2: 103  -> 3
#         # 3: 66   -> 4
#         # 6: 29   -> 5
#         # 5: 23   -> 6
#         remap = {
#             1: 0,
#             0: 1,
#             4: 2,
#             2: 3,
#             3: 4,
#             6: 5,
#             5: 6,
#         }
#         self.targets = [remap[target] for target in self.targets]
#         self.num_samples_per_cls = [
#             self.targets.count(i) for i in range(self.num_classes)
#         ]
#         # train:[5364, 890, 879, 411, 261, 113, 92]
#         # test: [1341, 223, 220, 103, 66,  29,  23]
#         self.group_mode = "class"
#         if self.dual_sample:
#             self.class_weight, self.sum_weight = self.get_weight(
#                 self.get_annotations(), self.num_classes)
#             self.class_dict = self._get_class_dict()

#     def __getitem__(self, index):

#         img_path, target = self.img_paths[index], self.targets[index]
#         img = Image.open(img_path).convert('RGB')
#         # img = cv2.imread(img_path)
#         # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         meta = dict()

#         if self.dual_sample:
#             if self.dual_sample_type == "reverse":
#                 sample_class = self.sample_class_index_by_weight()
#                 sample_indexes = self.class_dict[sample_class]
#                 sample_index = random.choice(sample_indexes)
#             elif self.dual_sample_type == "balance":
#                 sample_class = random.randint(0, self.num_classes - 1)
#                 sample_indexes = self.class_dict[sample_class]
#                 sample_index = random.choice(sample_indexes)
#             elif self.dual_sample_type == "uniform":
#                 sample_index = random.randint(0, self.__len__() - 1)

#             sample_img, sample_label = self.img_paths[
#                 sample_index], self.targets[sample_index]
#             sample_img = Image.open(sample_img).convert('RGB')
#             sample_img = self.transform(sample_img)

#             meta['sample_image'] = sample_img
#             meta['sample_label'] = sample_label

#         if self.transform is not None:
#             mean, std = self.splitfold_mean_std[self.fold_i]
#             img = self.transform(img, mean=mean, std=std)

#         return img, target, meta

#     def __len__(self):
#         return len(self.targets)

#     def get_data(self, fold_i, root, phase):
#         """fetch the img names and targets"""

#         fold_i += 1
#         csv_fpath = join(root, "split_data", "origin_split_data",
#                          f"split_data_{fold_i}_fold_{phase}.csv")
#         dataframe = pd.read_csv(csv_fpath)
#         img_names = list(dataframe.iloc[:, 0])
#         targets = list(dataframe.iloc[:, 1])

#         return img_names, targets

#     def get_annotations(self):
#         annos = []
#         for target in self.targets:
#             annos.append({'category_id': int(target)})
#         return annos

#     def _get_class_dict(self):
#         class_dict = dict()
#         for i, anno in enumerate(self.get_annotations()):
#             cat_id = anno["category_id"]
#             if cat_id not in class_dict:
#                 class_dict[cat_id] = []
#             class_dict[cat_id].append(i)
#         return class_dict

#     def get_weight(self, annotations, num_classes):
#         num_list = [0] * num_classes
#         cat_list = []
#         for anno in annotations:
#             category_id = anno["category_id"]
#             num_list[category_id] += 1
#             cat_list.append(category_id)
#         max_num = max(num_list)
#         class_weight = [max_num / i for i in num_list]
#         sum_weight = sum(class_weight)
#         return class_weight, sum_weight

#     def sample_class_index_by_weight(self):
#         rand_number, now_sum = random.random() * self.sum_weight, 0
#         for i in range(self.num_classes):
#             now_sum += self.class_weight[i]
#             if rand_number <= now_sum:
#                 return i


@Datasets.register_module("FGVC_BBN")
class FGVC_BBN(torch.utils.data.Dataset):
    """Original image size: (3, 450, 600)"""
    num_classes = 7
    splitfold_mean_std = [
        ([0.5709, 0.5464, 0.7634], [0.1701, 0.1528, 0.1408]),
        ([0.5707, 0.5462, 0.7634], [0.1707, 0.1531, 0.1416]),
        ([0.5704, 0.5462, 0.7642], [0.1704, 0.1528, 0.1410]),
        ([0.5701, 0.5458, 0.7636], [0.1702, 0.1530, 0.1413]),
        ([0.5705, 0.5461, 0.7629], [0.1703, 0.1528, 0.1413]),
    ]

    def __init__(self,
                 phase,
                 dual_sample=False,
                 dual_sample_type=None,
                 root="Skin7",
                 fold_i=0,
                 transform=None,
                 **kwargs):
        # phase: "train" or "test"
        # fold_i: one of [0, 1, 2, 3, 4]

        if "/" not in root:
            root = join(DATASETS_ROOT, root)

        self.phase = phase
        self.fold_i = fold_i
        self.transform = transform

        data_dir = join(root, 'ISIC2018_Task3_Training_Input')
        self.img_names, self.targets = self.get_data(fold_i, root, phase)
        self.dual_sample = True if dual_sample and phase == 'train' else False
        self.dual_sample_type = dual_sample_type
        self.img_paths = [
            join(data_dir, img_name) for img_name in self.img_names
        ]

        # 1: 1341 -> 0
        # 0: 223  -> 1
        # 4: 220  -> 2
        # 2: 103  -> 3
        # 3: 66   -> 4
        # 6: 29   -> 5
        # 5: 23   -> 6
        remap = {
            1: 0,
            0: 1,
            4: 2,
            2: 3,
            3: 4,
            6: 5,
            5: 6,
        }
        self.targets = [remap[target] for target in self.targets]
        self.num_samples_per_cls = [
            self.targets.count(i) for i in range(self.num_classes)
        ]
        # train:[5364, 890, 879, 411, 261, 113, 92]
        # test: [1341, 223, 220, 103, 66,  29,  23]
        self.group_mode = "class"
        if self.dual_sample:
            self.class_weight, self.sum_weight = self.get_weight(
                self.get_annotations(), self.num_classes)
            self.class_dict = self._get_class_dict()

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    def get_data(self, fold_i, root, phase):
        """fetch the img names and targets"""

        fold_i += 1
        csv_fpath = join(root, "split_data", "origin_split_data",
                         f"split_data_{fold_i}_fold_{phase}.csv")
        dataframe = pd.read_csv(csv_fpath)
        img_names = list(dataframe.iloc[:, 0])
        targets = list(dataframe.iloc[:, 1])

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


if __name__ == '__main__':
    print()
