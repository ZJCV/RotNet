# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午8:44
@file: crossentropy.py
@author: zj
@description: 
"""

import torch.nn as nn
from rotnet.model import registry


@registry.CRITERIONS.register('crossentropy')
def build_crossentropy(cfg):
    return nn.CrossEntropyLoss()
