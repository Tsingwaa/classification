from utils.core import Registry

Backbones = Registry("backbone")


def build_backbone(backbone_name, **kwargs):
    try:
        backbone = Backbones.get(backbone_name)
        return backbone(**kwargs)
    except Exception as error:
        print(f"model build fail : {error}")
        return None
