# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午8:50
@file: step_lr.py
@author: zj
@description: 
"""

import torch.optim as optim
from torch.optim.optimizer import Optimizer

from rotnet.optim import registry


@registry.LR_SCHEDULERS.register('step_lr')
def build_step_lr(cfg, optimizer):
    assert isinstance(optimizer, Optimizer)

    step_size = cfg.LR_SCHEDULER.STEP_SIZE
    gamma = cfg.LR_SCHEDULER.GAMMA

    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
