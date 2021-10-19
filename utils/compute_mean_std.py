import numpy as np
from PIL import Image
from pudb import set_trace
import torchvision
from tqdm import tqdm


def compute_mean_and_std(dataset):
    # 输入PyTorch的dataset，输出均值和标准差

    mean_r = 0.
    mean_g = 0.
    mean_b = 0.
    print("===> Computing mean...")
    for img_path, _ in tqdm(dataset, ncols=80):
        img = Image.open(img_path)
        img = np.asarray(img)  # change PIL Image to numpy array
        if len(img.shape) < 3 or img.shape[2] < 3:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0.
    diff_g = 0.
    diff_b = 0.

    N = 0.
    print("===> Computing std...")
    for img_path, _ in tqdm(dataset, ncols=80):
        img = Image.open(img_path)
        img = np.asarray(img)
        if len(img.shape) < 3 or img.shape[2] < 3:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0,
            mean_g.item() / 255.0,
            mean_r.item() / 255.0)
    std = (std_b.item() / 255.0,
           std_g.item() / 255.0,
           std_r.item() / 255.0)

    return mean, std


if __name__ == "__main__":
    data_root = "/home/waa/Data/miniImageNet"
    train_root = data_root + "/train"
    # val_root = data_root + '/val'
    # test_root = data_root + "/test"
    trainset = torchvision.datasets.ImageFolder(train_root)
    # valset = torchvision.datasets.ImageFolder(val_root)
    # testset = torchvision.datasets.ImageFolder(test_root)

    train_mean, train_std = compute_mean_and_std(trainset.imgs)

    print("训练集的平均值：{}，方差：{}".format(train_mean, train_std))
