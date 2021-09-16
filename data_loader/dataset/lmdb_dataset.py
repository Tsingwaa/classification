#coding=utf-8
import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import six
import string

import lmdb
import pickle
import msgpack
import tqdm

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import cv2
import random
import numpy as np
from common.dataset.builder import Datasets

@Datasets.register_module("lmdbdataset")
class ImageFolderLMDB(data.Dataset):

    def __init__(
            self,
            data_root=None,
            img_lst_fpath=None,
            map_fpath=None,
            transform=None,
            use_cv2=False,
            use_aug=False,
            use_balanced_aug=False,
            global_rank=0,
            global_size=1,
            **kwargs):
        self.db_path = data_root
        self.use_cv2 = use_cv2
        self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode()))

            self.labels = []
            for index in range(self.length):
                label_index = 'label-%09d' % (index + 1)
                self.labels.append(int(txn.get(label_index.encode())))
            assert len(self.labels) == self.length

        self.transform = transform
        print('Initialized LMDB dataset.')

    def __getitem__(self, index):
        img_index, label_index = 'image-%09d' % (index+1), 'label-%09d' % (index+1)
        img, label = None, None
        env = self.env
        with env.begin(write=False) as txn:
            # load label
            label = int(txn.get(label_index.encode()))

            # load image
            img = txn.get(img_index.encode())
            if self.use_cv2:
                img = cv2.imdecode(np.frombuffer(img, np.uint8), 3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # PIL
                buf = six.BytesIO()
                buf.write(img)
                buf.seek(0)
                img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
    """
    @property
    def labels(self):
        return self.labels
    """


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(dpath, name="train", write_frequency=5000, num_workers=16):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        # print(type(data), data)
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

if __name__ == '__main__':
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=20)

    args = parser.parse_args()

    folder2lmdb(args.folder, num_workers=args.procs, name=args.split)
    """
