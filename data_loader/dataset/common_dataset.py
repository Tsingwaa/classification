from collections import Counter
import os.path as P
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
import os
import json
from PIL import Image
from PIL import ImageFile
from data_loader.dataset.builder import Datasets

ImageFile.LOAD_TRUNCATED_IMAGES = True


@Datasets.register_module("testdataset")
class TestDataset(Dataset):
    def __init__(
            self,
            data_root=None,
            img_lst_fpath=None,
            map_fpath=None,
            transform=None,
            **kwarg):
        self.data_root = data_root
        self.img_lst_fpath = img_lst_fpath
        self.map_fpath = map_fpath
        self.transform = transform

        # generate label2ctg map
        label2ctg = {}
        with open(self.map_fpath, 'r') as f:
            for line in f.readlines():
                ctg, label = line.strip().split('\t')
                label2ctg[int(label)] = ctg

        # read images filename list
        self.img_fnames = []
        with open(self.img_lst_fpath, 'r') as fp:
            d = json.load(fp)
            img_urls = d['背面']
            img_urls.extend(d['正面'])
            for url in img_urls:
                fname = os.path.split(url)[1]
                self.img_fnames.append(fname)

        print(f'Loading test dataset: {len(self.img_fnames)}.')

    def __getitem__(self, index):
        img_fpath = P.join(self.data_root, self.img_fnames[index])
        img_fname = self.img_fnames[index]

        img = Image.open(img_fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_fname

    def __len__(self):
        return len(self.img_fnames)


@Datasets.register_module("evaldataset")
class EvalDataset(Dataset):
    def __init__(
            self,
            data_root=None,
            img_lst_fpath=None,
            map_fpath=None,
            transform=None,
            **kwarg):
        self.data_root = data_root
        self.transform = transform

        # reading img file from file
        label_map = {}
        with open(map_fpath) as fm:
            for line in fm.readlines():
                ctg, label = line.strip().split('\t')
                label_map[ctg] = int(label)

        print("Preparing val image datasets...")
        self.img_fnames = []
        self.labels = []
        fp = open(img_lst_fpath, 'r')
        for line in fp.readlines():
            fname, ctg = line.strip().split('\t')
            label = label_map[ctg]
            self.img_fnames.append(fname)
            self.labels.append(label)

        fp.close()

        self.img_fnames = np.array(self.img_fnames)
        self.labels = np.array(self.labels)  # .reshape(-1, 1)

        label_cou = Counter(self.labels)
        label_set = sorted(list(set(self.labels)))
        self.cls_num_list = [label_cou[lab] for lab in label_set]

    def __getitem__(self, index):
        img_fpath = P.join(self.data_root, self.img_fnames[index])
        img_fname = self.img_fnames[index]
        img = Image.open(img_fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(np.array(self.labels[index]))

        return img, label, img_fname

    def __len__(self):
        return len(self.labels)


@Datasets.register_module("imagedataset")
class ImageDataset(Dataset):
    def __init__(
            self,
            data_root=None,
            img_lst_fpath=None,
            map_fpath=None,
            transform=None,
            **kwargs):
        self.data_root = data_root
        self.transform = transform

        # reading img file from file
        label_map = {}
        with open(map_fpath) as fm:
            for line in fm.readlines():
                ctg, label = line.strip().split('\t')
                label_map[ctg] = int(label)

        # print("Preparing image datasets...")
        self.img_fnames = []
        self.labels = []
        fp = open(img_lst_fpath, 'r')
        for line in fp.readlines():
            fname, ctg = line.strip().split('\t')
            label = label_map[ctg]
            '''
            if not P.exists(P.join(self.img_path, filename)):
                #print('file not exist', P.join(self.img_path, filename))
                continue
            '''
            self.img_fnames.append(fname)
            self.labels.append(label)
        fp.close()

        self.img_fnames = np.array(self.img_fnames)
        self.labels = np.array(self.labels)  # .reshape(-1, 1)
        label_cou = Counter(self.labels)
        label_set = sorted(list(set(self.labels)))
        self.cls_num_list = [label_cou[lab] for lab in label_set]

    def __getitem__(self, index):
        img_fname = self.img_fnames[index]
        img_fpath = P.join(self.data_root, img_fname)
        img = Image.open(img_fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(np.array(self.labels[index]))

        return img, label

    def __len__(self):
        return len(self.img_fnames)


@Datasets.register_module("imagedataset_multi_label")
class ImageMultilabelDataset(Dataset):
    def __init__(
            self,
            data_root=None,
            img_lst_fpath=None,
            map_fpath=None,
            transform=None,
            **kwargs):
        self.data_root = data_root
        self.transform = transform

        self.img_fnames = []
        self.labels = []
        fp = open(img_lst_fpath, 'r')
        for line in fp.readlines():
            # format: 'img lab0 lab1 ... labN'
            r = line.strip().split('\t')
            fname = r[0]
            label = [float(it) for it in r[1:]]
            '''
            if not P.exists(P.join(self.img_path, filename)):
                print(P.join(self.img_path, filename))
                continue
            '''
            self.img_fnames.append(fname)
            self.labels.append(label)
        fp.close()
        self.img_fname = np.array(self.img_fname)
        self.labels = np.array(self.labels)  # .reshape(-1, 1)

    def __getitem__(self, index):
        img_fpath = P.join(self.data_root, self.img_fname[index])
        img = Image.open(img_fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[index])
        return img, label

    def __len__(self):
        return len(self.img_fnames)
