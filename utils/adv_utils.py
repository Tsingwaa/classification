from functools import partial


def switch_mode(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


switch_clean = partial(switch_mode, status='clean')
switch_adv = partial(switch_mode, status='adv')
switch_mix = partial(switch_mode, status='mix')
