from functools import partial


def switch_mode(m, status):
    if hasattr(m, 'is_clean'):
        m.is_clean = status


switch_clean = partial(switch_mode, status=True)
switch_adv = partial(switch_mode, status=False)
