import os

import pandas as pd
import torch
from data_loader.dataset.builder import Datasets
from PIL import Image


def make_dataset(fold, data_root):
    train_csv_fname = 'split_data/origin_split_data/\
            split_data_{}_fold_train.csv'.format(fold)
    train_csv_fpath = os.path.join(data_root, train_csv_fname)
    train_dataframe = pd.read_csv(train_csv_fpath)
    raw_train_data = train_dataframe.values
    train_data = []

    for x, y in raw_train_data:
        train_data.append((x, y))

    test_csv_fname = 'split_data/origin_split_data/\
            split_data_{}_fold_test.csv'.format(fold)
    test_csv_fpath = os.path.join(data_root, test_csv_fname)
    test_dataframe = pd.read_csv(test_csv_fpath)
    raw_test_data = test_dataframe.values
    test_data = []

    for x, y in raw_test_data:
        test_data.append((x, y))

    return train_data, test_data


@Datasets.register_module("Skin7")
class Skin7(torch.utils.data.Dataset):
    ''' 原图大小（3， 450， 600） '''
    num_classes = 7
    mean = [0.7626, 0.5453, 0.5714]
    std = [0.1404, 0.1519, 0.1685]

    def __init__(self, data_root, train, transform=None, fold=0, **kwargs):
        self.train_data, self.test_data = make_dataset(fold, data_root)
        print('===> Initializing fold.{} {}...\
              '.format(fold, 'train set' if train else 'test set'))

        self.train = train

        if self.train:
            self.labels = [data[1] for data in self.train_data]
            self.img_num_per_cls = [
                self.labels.count(i) for i in range(self.num_classes)
            ]
        else:
            self.labels = [data[1] for data in self.test_data]

        self.transform = transform

        raw_train_data = 'ISIC2018_Task3_Training_Input'
        self.data_dir = os.path.join(data_root, raw_train_data)

    def __getitem__(self, index):
        img_fname, label = self.train_data[index] if self.train \
            else self.test_data[index]
        img_fpath = os.path.join(self.data_dir, img_fname)
        img = Image.open(img_fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)


def get_mean_std(fold):
    filename = './mean_std.csv'
    dataframe = pd.read_csv(filename).values[int(fold) - 1]
    print(dataframe)

    return dataframe[0:3], dataframe[3:]


if __name__ == '__main__':
    '''
    计算每类数据量
    num_dict = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    train_data, test_data = make_dataset(1,"/data/Public/Datasets/Skin7")
    target = [tup[1] for tup in train_data]
    # print(test_data)
    # print(len(train_data) ,len(test_data))
    target.extend([tup[1] for tup in test_data])
    class_count = [target.count(i) for i in range(len(num_dict))]
    print(sum(class_count), len(target))
    if sum(class_count) == len(target):
        class_dict = {num_dict[i]:class_count[i] for i in range(len(num_dict))}
        print(class_dict)

        # {'MEL':1113, 'NV':6705, 'BCC':514, 'AKIEC':327,
        'BKL':1099, 'DF':115, 'VASC':142}

    '''
    # get_mean_std(1)
