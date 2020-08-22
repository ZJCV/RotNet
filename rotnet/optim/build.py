# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:55
@file: build.py
@author: zj
@description: 
"""

import torch.optim as optim


def build_optimizer(model):
    return optim.Adam(model.parameters(), lr=3e-4, weight_decay=3e-5)


def build_lr_scheduler(optimizer):
    return optim.lr_scheduler.StepLR(optimizer, step_size=10)
