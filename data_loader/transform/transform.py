import torchvision.transforms as transforms
from .randaugment import RandAugment
from .randaugment_fixmatch import RandAugmentMC

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def base_transform(resize=(224, 224), phase='train', **kwargs):
    if phase == 'train':
        # transforms.RandomOrder
        return transforms.Compose([
            transforms.Resize(int(resize[0] / 0.875)),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing()
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0]), int(resize[1]))),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def flower_transform(resize=(224, 224), phase='train', use_cv2=False, **kwargs):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(int(resize[0] / 0.875)),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing()
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0]), int(resize[1]))),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def huashu_transform(resize=(224, 224), phase='train', use_cv2=False, **kwargs):
    if phase == 'train':
        # transforms.RandomOrder
        return transforms.Compose([
            transforms.Resize(
                size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing()
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0]), int(resize[1]))),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def common_transform(resize=(224, 224), phase='train', use_cv2=False, **kwargs):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(int(resize[0] / 0.875)),
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
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0]), int(resize[1]))),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def rand_transform(resize=(224, 224), phase='train', **kwargs):
    if phase == "train":
        n = kwargs.get("rand_n", 2)
        m = kwargs.get("rand_m", 10)
        transform = transforms.Compose([transforms.Resize(int(resize[0] / 0.875)),
                                        transforms.RandomCrop(resize),
                                        RandAugment(n, m),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=(int(resize[0]), int(resize[1]))),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform


class TransformFixMatch(object):
    def __init__(self, resize=(224, 224), phase='train', **kwargs):
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
