import torch
from data_loader.transform.builder import Transforms
from torchvision import transforms

# from .randaugment import RandAugment
from .randaugment_fixmatch import RandAugmentMC, RandAugmentPC

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]


@Transforms.register_module('BaseTransform')
class BaseTransform:

    def __init__(self,
                 phase='train',
                 resize=(224, 224),
                 strong=False,
                 **kwargs):
        self.phase = phase
        self.resize = resize
        self.strong = strong

        if phase == 'train':
            self.insert_T = kwargs.get('insert_T', None)

    def __call__(self, x, mean=IN_MEAN, std=IN_STD, **kwargs):
        if self.phase == 'train':
            if self.strong:
                ret_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    # transforms.RandomVerticalFlip(0.5),
                    transforms.RandomAffine(
                        degrees=45,
                        translate=(0.4, 0.4),
                        scale=(0.5, 1.5),
                        shear=20,
                        fill=(127, 127, 127),
                    ),
                    transforms.RandomResizedCrop(self.resize),
                    # transforms.ColorJitter(brightness=0.4,
                    #                        saturation=0.4,
                    #                        contrast=0.4,
                    #                        hue=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    # transforms.RandomErasing(),
                ])
            else:
                # T_list = [transforms.Resize(self.resize),
                #           transforms.ToTensor(),
                #           transforms.Normalize(mean, std), ]

                # if self.insert_T == 'rot':
                #     T_list.insert(0, transforms.RandomRotation(30))
                # elif self.insert_T == 'hlip':
                #     T_list.insert(0, transforms.RandomHorizontalFlip(0.5))
                # elif self.insert_T == 'vflip':
                #     T_list.insert(0, transforms.RandomVerticalFlip(0.5))
                # elif self.insert_T == 'cj':
                #     T_list.insert(0, transforms.ColorJitter(brightness=0.1,
                #                                             saturation=0.1,
                #                                             contrast=0.1,
                #                                             hue=0.1,))
                # elif self.insert_T == 'rscrop':
                #     T_list.insert(0,
                #                   transforms.RandomResizedCrop(self.resize))
                # elif self.insert_T == 'tsl':
                #     T_list.insert(0, transforms.RandomAffine(
                #         degrees=0,
                #         translate=(0.2, 0.2)))
                # elif self.insert_T == 'scale':
                #     T_list.insert(0, transforms.RandomAffine(
                #         degrees=0,
                #         scale=(0.75, 1.25)))
                # elif self.insert_T == 'shear':
                #     T_list.insert(0, transforms.RandomAffine(
                #         degrees=0,
                #         shear=10))
                # ret_transform = transforms.Compose(T_list)

                ret_transform = transforms.Compose([
                    # transforms.Resize(int(self.resize[0] / 0.875)),
                    # transforms.CenterCrop(self.resize),
                    # transforms.RandomRotation(30),
                    transforms.RandomAffine(
                        degrees=30,
                        translate=(0.2, 0.2),
                        scale=(0.75, 1.25),
                        shear=10,
                        # fill=(127, 127, 127),
                    ),
                    transforms.RandomResizedCrop(self.resize),
                    transforms.RandomHorizontalFlip(0.5),
                    # T.ColorJitter(brightness=0.05,
                    #               saturation=0.05,
                    #               contrast=0.05,
                    #               hue=0.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    # transforms.RandomErasing(),
                ])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        return ret_transform(x)


@Transforms.register_module('CifarTransform')
class CifarTransform:

    def __init__(self, phase='train', resize=(32, 32), strong=False, **kwargs):
        self.phase = phase
        self.resize = resize
        self.strong = strong

    def __call__(self, x, mean=IN_MEAN, std=IN_STD, **kwargs):
        if self.phase == 'train':
            if self.strong:
                ret_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    # transforms.RandomVerticalFlip(0.5),
                    transforms.RandomAffine(
                        degrees=45,
                        translate=(0.4, 0.4),
                        scale=(0.5, 1.5),
                        shear=20,
                        fill=(127, 127, 127),
                    ),
                    transforms.RandomResizedCrop(self.resize),
                    # transforms.ColorJitter(brightness=0.4,
                    #                        saturation=0.4,
                    #                        contrast=0.4,
                    #                        hue=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    # transforms.RandomErasing(),
                ])
            else:
                ret_transform = transforms.Compose([
                    transforms.RandomCrop(self.resize, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        return ret_transform(x)


@Transforms.register_module('MedTransform')
class MedTransform:

    def __init__(self, phase='train', resize=(224, 224), **kwargs):
        self.phase = phase
        self.resize = resize

    def __call__(self, x, mean=IN_MEAN, std=IN_STD, **kwargs):
        if self.phase == 'train':
            ret_transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.RandomCrop(self.resize),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        return ret_transform(x)


@Transforms.register_module('ImagenetTransform')
class ImagenetTransform:

    def __init__(self,
                 phase='train',
                 resize=(224, 224),
                 strong=False,
                 **kwargs):
        self.phase = phase
        self.resize = resize
        self.strong = strong

    def __call__(self, x, mean=IN_MEAN, std=IN_STD, **kwargs):
        if self.phase == 'train':
            ret_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.resize),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.4,
                #                        contrast=0.4,
                #                        saturation=0.4,
                #                        hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        return ret_transform(x)


@Transforms.register_module('NoiseBaseTransform')
class NoiseBaseTransform:

    def __init__(self,
                 phase='train',
                 resize=(224, 224),
                 strong=False,
                 sigma=0.005,
                 sigma_low=None,
                 sigma_high=None,
                 **kwargs):
        self.phase = phase
        self.resize = resize
        self.strong = strong
        self.sigma = sigma
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high

    def __call__(self, x, mean=IN_MEAN, std=IN_STD, percent=None):
        if self.phase == 'train':
            if not self.sigma and self.sigma_low:
                sigma_range = self.sigma_high - self.sigma_low
                self.sigma = self.sigma_low + percent * sigma_range

            if self.strong:
                ret_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    # transforms.RandomVerticalFlip(0.5),
                    transforms.RandomAffine(
                        degrees=45,
                        translate=(0.4, 0.4),
                        scale=(0.5, 1.5),
                        shear=30,
                        # fill=(127, 127, 127)
                    ),
                    transforms.RandomResizedCrop(self.resize),
                    # transforms.ColorJitter(brightness=0.4,
                    #                        saturation=0.4,
                    #                        contrast=0.4,
                    #                        hue=0.4),
                    transforms.ToTensor(),
                    GaussianNoise(self.sigma),
                    transforms.Normalize(mean, std),
                    # transforms.RandomErasing(),
                ])
            else:
                ret_transform = transforms.Compose([
                    transforms.RandomAffine(
                        degrees=30,
                        translate=(0.2, 0.2),
                        scale=(0.75, 1.25),
                        shear=10,
                        # fill=(127, 127, 127)
                    ),
                    transforms.RandomResizedCrop(self.resize),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    GaussianNoise(self.sigma),
                    transforms.Normalize(mean, std),
                    # transforms.RandomErasing(),
                ])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        return ret_transform(x)


# from ildoonet/pytorch-randaugment
@Transforms.register_module('RandTransform')
class RandTransform:

    def __init__(self, phase='train', resize=(32, 32), **kwargs):
        self.phase = phase
        self.resize = resize
        self.n = kwargs.get('rand_n', 2)
        self.m = kwargs.get('rand_m', 10)
        self.strong = kwargs.get('strong', False)

    def __call__(
        self,
        x,
        percent=None,
        m=None,
        n=None,
        mean=IN_MEAN,
        std=IN_STD,
    ):
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
                transforms.Normalize(mean, std)
            ])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize(int(self.resize[0] / 0.875)),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        return ret_transform(x)


@Transforms.register_module('AdvTransform')
class AdvTransform:

    def __init__(self,
                 phase='train',
                 resize=(224, 224),
                 strong=False,
                 **kwargs):
        self.phase = phase
        self.resize = resize
        self.strong = strong

    def __call__(self, x, mean=IN_MEAN, std=IN_STD, **kwargs):
        if self.phase == 'train':
            if self.strong:
                ret_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomAffine(
                        degrees=45,
                        translate=(0.4, 0.4),
                        scale=(0.5, 1.5),
                        shear=30,
                        fill=(127, 127, 127),
                    ),
                    transforms.RandomResizedCrop(self.resize),
                    transforms.ToTensor(),
                ])
            else:
                ret_transform = transforms.Compose([
                    transforms.RandomAffine(
                        degrees=30,
                        translate=(0.2, 0.2),
                        scale=(0.25, 1.25),
                        shear=10,
                    ),
                    transforms.RandomResizedCrop(self.resize),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                ])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize(self.resize),
                transforms.ToTensor(),
            ])

        return ret_transform(x)


@Transforms.register_module('AdvCifarTransform')
class AdvCifarTransform:

    def __init__(self, phase='train', resize=(32, 32), strong=False, **kwargs):
        self.phase = phase
        self.resize = resize
        self.strong = strong

    def __call__(self, x, mean=IN_MEAN, std=IN_STD, **kwargs):
        if self.phase == 'train':
            if self.strong:
                ret_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    # transforms.RandomAffine(
                    #     degrees=30,
                    #     translate=(0.4, 0.4),
                    #     scale=(0.5, 1.5),
                    #     shear=30,
                    #     fill=(127, 127, 127),
                    # ),
                    transforms.RandomResizedCrop(self.resize),
                    transforms.ToTensor(),
                ])
            else:
                ret_transform = transforms.Compose([
                    transforms.RandomCrop(self.resize, padding=4),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                ])
        else:
            ret_transform = transforms.Compose([
                transforms.Resize(self.resize),
                transforms.ToTensor(),
            ])

        return ret_transform(x)


@Transforms.register_module('TransformFixMatch')
class TransformFixMatch(object):

    def __init__(self,
                 phase='train',
                 resize=(224, 224),
                 mean=None,
                 std=None,
                 **kwargs):

        self.phase = phase
        n = kwargs.get('rand_n', 2)
        m = kwargs.get('rand_m', 10)
        self.is_labeled = kwargs.get('is_labeled', True)
        self.weakaug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(int(resize[0] / 0.875),
                                    int(resize[1] / 0.875))),
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
            transforms.Resize(size=(int(resize[0]), int(resize[1]))),
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


def common_transform(phase='train',
                     resize=(224, 224),
                     mean=None,
                     std=None,
                     **kwargs):

    if phase == 'train':
        ret_transform = transforms.Compose([
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
        ret_transform = transforms.Compose([
            transforms.Resize(resize),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return ret_transform


class GaussianNoise:

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        noise = self.sigma * torch.randn_like(img)
        noised_tensor = img + noise
        noised_tensor = torch.clamp(noised_tensor, min=0., max=1.)

        return noised_tensor
