from common.core import Registry

Backbones = Registry("backbones")


def build_backbone(backbone_name, **kwargs):
    try:
        backbone = Backbones.get(backbone_name)
        return backbone(**kwargs).get_model()
    except Exception as error:
        print(f"model build fail : {error}")
        return None
