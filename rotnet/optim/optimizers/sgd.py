# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午8:55
@file: sgd.py
@author: zj
@description: 
"""

import torch.nn as nn
import torch.optim as optim

from rotnet.optim import registry


@registry.OPTIMIZERS.register('sgd')
def build_sgd(cfg, model):
    assert isinstance(model, nn.Module)

    lr = cfg.OPTIMIZER.LR
    momentum = cfg.OPTIMIZER.MOMENTUM
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY

    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
