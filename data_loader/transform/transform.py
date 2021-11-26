import torchvision.transforms as transforms
from .randaugment_fixmatch import RandAugmentMC, RandAugmentPC
from .randaugment import RandAugment
from data_loader.transform.builder import Transforms

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]


@Transforms.register_module('BaseTransform')
class BaseTransform:
    def __init__(self, phase='train', resize=(224, 224), **kwargs):
        self.phase = phase
        self.resize = resize

    def __call__(self, x, mean=IN_MEAN, std=IN_STD):
        if self.phase == 'train':
            ret_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.resize),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(25),
                transforms.ColorJitter(
                    brightness=0.4,
                    saturation=0.4,
                    contrast=0.4,
                    hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                # transforms.RandomErasing()
            ])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize(int(self.resize[1] / 0.875)),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        return ret_transform(x)


# from ildoonet/pytorch-randaugment
@Transforms.register_module('RandTransform')
class RandTransform:
    def __init__(self, phase='train', resize=(32, 32),
                 **kwargs):
        self.phase = phase
        self.resize = resize
        self.n = kwargs.get('rand_n', 2)
        self.m = kwargs.get('rand_m', 10)
        self.strong = kwargs.get('strong', False)

    def __call__(self, x, percent=None, m=None, n=None,
                 mean=IN_MEAN, std=IN_STD,):
        if m is None:
            m = self.m
        if n is None:
            n = self.n
        if percent is not None:
            m = int(((1 + percent) / 2 * m))

        if self.phase == 'train':
            ret_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.resize),
                RandAugmentPC(n, m) if self.strong else RandAugmentMC(n, m),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize(int(self.resize[0] / 0.875)),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        return ret_transform(x)


@Transforms.register_module('AdvTransform')
class AdvTransform:
    def __init__(self, phase='train', resize=(224, 224), **kwargs):
        self.phase = phase
        self.resize = resize

    def __call__(self, x, mean=IN_MEAN, std=IN_STD):
        if self.phase == 'train':
            ret_transform = transforms.Compose([
                transforms.Resize(int(self.resize[1] / 0.875)),
                transforms.RandomCrop(self.resize),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(25),
                # transforms.ColorJitter(
                #     brightness=0.4,
                #     saturation=0.4,
                #     contrast=0.4,
                #     hue=0.05
                # ),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
                # transforms.RandomErasing()
            ])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize(int(self.resize[1] / 0.875)),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
            ])
        return ret_transform(x)


@Transforms.register_module('TransformFixMatch')
class TransformFixMatch(object):
    def __init__(self, phase='train', resize=(224, 224),
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


def flower_transform(resize=(224, 224), phase='train',
                     mean=None, std=None, **kwargs):
    if phase == 'train':
        ret_transform = transforms.Compose([
            transforms.Resize(int(resize[1] / 0.875)),
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
            transforms.RandomCrop(resize[1], padding=4),
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


# def cifar_adaptive_transform(phase='train',
#                              resize=(32, 32),
#                              cls=0, **kwargs):
#     mean = [0.4914, 0.4822, 0.4465]
#     std = [0.2023, 0.1994, 0.2010]

#     if phase == 'train':
#         ret_transform = transforms.Compose([
#             transforms.RandomCrop(resize[1], padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#         ])
#     else:
#         ret_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])
#     return ret_transform
