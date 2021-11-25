"""Utility for Network"""


def freeze_model(model, unfreeze_keys=['fc']):
    for k, v in model.named_parameters():
        if any(key in k for key in unfreeze_keys):
            v.required_grad = True
        else:
            v.required_grad = False

    return model
