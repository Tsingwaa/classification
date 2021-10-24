from utils.core import Registry

Samplers = Registry("sampler")


def build_sampler(sampler_name, **kwargs):
    try:
        sampler_template = Samplers.get(sampler_name)
        sampler = sampler_template(**kwargs)
        return sampler
    except Exception as error:
        raise RuntimeError(f"sampler build failed {error}")
