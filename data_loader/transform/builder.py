from utils.core import Registry

Transforms = Registry("transform")


def build_transform(transform_type, **kwargs):
    return Transforms.get(transform_type)(**kwargs)
