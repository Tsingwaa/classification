from utils.core import Registry

Modules = Registry("module")


def build_module(module_name, **kwargs):
    try:
        module = Modules.get(module_name)
        return module(**kwargs)
    except Exception as error:
        print(f"Module build fail: {error}")
        return None
