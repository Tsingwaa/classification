import os

from core import Registry

Datasets = Registry("dataset")

DATASETS_ROOT = os.path.join(os.environ["HOME"], "Datasets")
