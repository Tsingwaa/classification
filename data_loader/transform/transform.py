import torchvision.transforms as transforms
from .randaugment import RandAugment
from .randaugment_fixmatch import RandAugmentMC

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]


def base_transform(phase='train', resize=(224, 224),
                   mean=None, std=None, **kwargs):
    if phase == 'train':
        # transforms.RandomOrder
        ret_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875),
                                    int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing()
        ])
    else:
        ret_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0]),
                                    int(resize[1]))),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return ret_transform


def flower_transform(resize=(224, 224), phase='train',
                     mean=None, std=None, **kwargs):
    if phase == 'train':
        ret_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875),
                                    int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing()
        ])
    else:
        ret_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0]),
                                    int(resize[1]))),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return ret_transform


def huashu_transform(phase='train', resize=(224, 224),
                     mean=None, std=None, **kwargs):
    if phase == 'train':
        # transforms.RandomOrder
        ret_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875),
                                    int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing()
        ])
    else:
        ret_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0]),
                                    int(resize[1]))),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return ret_transform


def common_transform(phase='train', resize=(224, 224),
                     mean=None, std=None, **kwargs):
    if phase == 'train':
        ret_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875),
                                    int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.RandomAffine(degrees=(30, 70),
                                    translate=(0.1, 0.3),
                                    scale=(0.5, 1.25)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing()
        ])
    else:
        ret_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0]),
                                    int(resize[1]))),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return ret_transform


def rand_transform(phase='train', resize=(224, 224),
                   mean=None, std=None, **kwargs):
    if phase == "train":
        n = kwargs.get("rand_n", 2)
        m = kwargs.get("rand_m", 10)
        ret_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875),
                                    int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            RandAugment(n, m),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        ret_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0]),
                                    int(resize[1]))),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return ret_transform


def cifar_transform(phase='train', resize=(32, 32), **kwargs):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    if phase == 'train':
        ret_transform = transforms.Compose([
            transforms.RandomCrop(resize, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        ret_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return ret_transform


class TransformFixMatch(object):
    def __init__(self, resize=(224, 224), phase='train',
                 mean=None, std=None, **kwargs):

        self.phase = phase
        n = kwargs.get('rand_n', 2)
        m = kwargs.get('rand_m', 10)
        self.is_labeled = kwargs.get('is_labeled', True)
        self.weakaug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(
                size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.strongaug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(resize),
            RandAugmentMC(n, m),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0]),
                                    int(resize[1]))),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __call__(self, x):
        if self.phase == 'train':
            if self.is_labeled:
                return self.weakaug(x)
            else:
                return self.weakaug(x), self.strongaug(x)
        else:
            return self.val_transform(x)
