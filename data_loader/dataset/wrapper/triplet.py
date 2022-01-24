import numpy as np
from data_loader.dataset.builder import Datasets
from PIL import Image
from torch.utils.data import Dataset


@Datasets.register_module("TripletWrapper")
class TripletWrapper(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative
    samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {
                label: np.where(self.train_labels.numpy() == label)[0]

                for label in self.labels_set
            }

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]

                for label in self.labels_set
            }

            random_state = np.random.RandomState(29)

            triplets = [[
                i,
                random_state.choice(
                    self.label_to_indices[self.test_labels[i].item()]),
                random_state.choice(self.label_to_indices[np.random.choice(
                    list(self.labels_set -
                         set([self.test_labels[i].item()])))])
            ] for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[
                index].item()
            positive_index = index

            while positive_index == index:
                positive_index = np.random.choice(
                    self.label_to_indices[label1])
            negative_label = np.random.choice(
                list(self.labels_set - set([label1])))
            negative_index = np.random.choice(
                self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)
