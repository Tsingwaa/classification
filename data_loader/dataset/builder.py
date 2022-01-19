import os

from utils.core import Registry

Datasets = Registry("dataset")

DATASETS_ROOT = os.path.join(os.environ["HOME"], "Datasets")
