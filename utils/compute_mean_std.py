###############################################################################
# Copyright (C) 2021 All rights reserved.
# Filename: compute_mean_std.py
# Author: Tsingwaa
# Email: zengchh3@gmail.com
# Created Time : 2021-12-20 19:27 Monday
# Last modified: 2021-12-20 19:27 Monday
# Description:
#
###############################################################################
# from os.path import join

import numpy as np
from data_loader.dataset.builder import Datasets
from PIL import Image
from tqdm import tqdm


def compute_mean_and_std(img_paths):
    # 输入PyTorch的数据路径列表，输出均值和标准差

    mean_r = 0.
    mean_g = 0.
    mean_b = 0.

    for img_path in tqdm(img_paths, ncols=80, desc="Computing mean"):
        img = Image.open(img_path)  # 默认打开的通道为BGR
        img = np.asarray(img)  # change PIL Image to numpy array

        if len(img.shape) < 3 or img.shape[2] < 3:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(img_paths)
    mean_g /= len(img_paths)
    mean_r /= len(img_paths)

    mean = (mean_r.item() / 255.0, mean_g.item() / 255.0,
            mean_b.item() / 255.0)

    print('===> mean: [{:.4f},{:.4f},{:.4f}]'.format(mean[0], mean[1],
                                                     mean[2]))

    diff_r = 0.
    diff_g = 0.
    diff_b = 0.

    N = 0.

    for img_path in tqdm(img_paths, ncols=80, desc="Computing std"):
        img = Image.open(img_path)  # 默认打开的通道为BGR
        img = np.asarray(img)

        if len(img.shape) < 3 or img.shape[2] < 3:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_r = np.sqrt(diff_r / N)
    std_g = np.sqrt(diff_g / N)
    std_b = np.sqrt(diff_b / N)

    std = (std_r.item() / 255.0, std_g.item() / 255.0, std_b.item() / 255.0)

    print('===> std: [{:.4f},{:.4f},{:.4f}]\n'.format(std[0], std[1], std[2]))

    return mean, std


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    dataset_name = "Xray14"
    root = "Xray14"

    for fold_i in range(5):
        print(f"===> Processing {dataset_name} fold-{fold_i}...")
        trainset = Datasets.get(dataset_name)(
            root=root,
            phase="train",
            fold_i=fold_i,
        )
        train_mean, train_std = compute_mean_and_std(trainset.img_paths)

    print('All is done.')
