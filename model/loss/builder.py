from common.core import Registry

Losses = Registry("Losses")


def build_loss(loss_name, **kwargs):
    try:
        return Losses.get(loss_name)(**kwargs)
    except Exception as error:
        print(f"loss build error {error}")
        return None
