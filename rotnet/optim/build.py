# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:55
@file: trainer.py
@author: zj
@description: 
"""

import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from . import registry
from .lr_schedulers.step_lr import build_step_lr
from .optimizers.sgd import build_sgd
from .optimizers.adam import build_adam


def build_optimizer(cfg, model):
    assert isinstance(model, nn.Module)
    return registry.OPTIMIZERS[cfg.OPTIMIZER.NAME](cfg, model)


def build_lr_scheduler(cfg, optimizer):
    assert isinstance(optimizer, Optimizer)
    return registry.LR_SCHEDULERS[cfg.LR_SCHEDULER.NAME](cfg, optimizer)
