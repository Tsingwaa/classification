import numpy as np
from data_loader.dataset.builder import Datasets
from PIL import Image
from torch.utils.data import Dataset


@Datasets.register_module("SiameseDataset")
class SiameseDataset(Dataset):
    """
    Train: For each sample, randomly creates a positive or a negative pair
    Test: Fast forward
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_targets = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.targets_set = set(self.train_targets.numpy())
            self.label_to_indices = {
                target: np.where(self.train_targets.numpy() == target)[0]

                for target in self.targets_set
            }
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.targets_set = set(self.test_labels.numpy())
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]

                for label in self.targets_set
            }

            random_state = np.random.RandomState(29)

            positive_pairs = [[
                i,
                random_state.choice(
                    self.label_to_indices[self.test_labels[i].item()]), 1
            ] for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[
                i,
                random_state.choice(self.label_to_indices[np.random.choice(
                    list(self.targets_set -
                         set([self.test_labels[i].item()])))]), 0
            ] for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_targets[
                index].item()

            if target == 1:
                siamese_index = index

                while siamese_index == index:
                    siamese_index = np.random.choice(
                        self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(
                    list(self.targets_set - set([label1])))
                siamese_index = np.random.choice(
                    self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)
