from utils.core import Registry

Samplers = Registry("sampler")


def build_sampler(sampler_name, **kwargs):
    try:
        sampler = Samplers.get(sampler_name)(**kwargs)
        return sampler
    except Exception as error:
        raise RuntimeError(f"sampler build failed {error}")
