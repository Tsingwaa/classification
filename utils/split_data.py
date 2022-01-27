###############################################################################
# Copyright (C) 2022 All rights reserved.
# Filename: fetch_filename_list.py
# Author: Tsingwaa
# Email: zengchh3@gmail.com
# Created Time : 2022-01-25 21:49 Tuesday
# Last modified: 2022-01-25 21:49 Tuesday
# Description: Split Imagefolder into 5-fold train-test.
#
###############################################################################

import os
import os.path as osp


def split_data(
    data_root,
    label2fnames_path,
    label2class_path,
    split_data_path,
):
    """split dataset
    - Label is created by cardinality in decreasing order
    - split into 5-fold

    Args:
        root: the root directory of data.
    """

    print("Getting filenames from root directory...")

    class_names = os.listdir(data_root)
    class2fnames = dict()
    class2fnum = dict()

    for class_name in class_names:
        dir_path = osp.join(data_root, class_name)
        fold_fnames = [
            osp.join(class_name, fname) for fname in os.listdir(dir_path)

            if osp.splitext(fname)[1] in ['.jpeg', '.jpg']
        ]
        class2fnames[class_name] = fold_fnames
        class2fnum[class_name] = len(fold_fnames)

    sorted_class2fnum = sorted(class2fnum.items(),
                               key=lambda kv: kv[1],
                               reverse=True)

    label2fnames = dict()
    label2class = dict()

    for i, (class_name, fnum) in enumerate(sorted_class2fnum):
        print(f"{i:>2}: {class_name:>4} : {fnum:>5}")
        label2fnames[i] = class2fnames[class_name]
        label2class[i] = class_name

    print(f"Writing label2fnames into '{label2fnames_path}'...")
    yaml_dump(label2fnames, label2fnames_path)

    print(f"Writing label2class into '{label2class_path}'...")
    yaml_dump(label2class, label2class_path)

    print("Splitting 5-fold data...")
    data_5fold = {i: [] for i in range(5)}  # store all splits.

    for i in range(5):
        for label, fold_fnames in label2fnames.items():
            fold_fnames = fold_fnames[i::5]
            fold_data = [[fname, label] for fname in fold_fnames]
            data_5fold[i].extend(fold_data)

    print(f"Writing split_data into {split_data_path}...")
    yaml_dump(data_5fold, split_data_path)


def json_dump(data, path):
    path += '.json'
    import json
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=True, indent=4)


def yaml_dump(data, path):
    path += ".yaml"
    import yaml
    with open(path, 'w') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


if __name__ == "__main__":
    data_root = osp.expanduser("~/Datasets/ISIC2019/Data")
    label2fnames_path = osp.expanduser("~/Datasets/ISIC2019/label2fnames")
    label2class_path = osp.expanduser("~/Datasets/ISIC2019/label2class")
    split_data_path = osp.expanduser("~/Datasets/ISIC2019/split_data")

    split_data(
        data_root=data_root,
        label2fnames_path=label2fnames_path,
        label2class_path=label2class_path,
        split_data_path=split_data_path,
    )
