from utils.core import Registry

Transforms = Registry("transform")


def build_transform(transform_name, **kwargs):
    try:
        transform = Transforms.get(transform_name)(**kwargs)
        return transform
    except Exception as error:
        raise RuntimeError(f"transform build failed {error}")
