from common.core import Registry

Transforms = Registry("Transform")


def build_transform(transform_type, **kwargs):
    return Transforms.get(transform_type)(**kwargs)
