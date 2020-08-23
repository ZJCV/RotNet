# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:30
@file: trainer.py
@author: zj
@description: 
"""

from . import registry
from .models.mobilenet_v2 import build_mobilenet_v2
from .criterions.crossentropy import build_crossentropy


def build_model(cfg):
    return registry.BACKBONES[cfg.MODEL.NAME](cfg)


def build_criterion(cfg):
    return registry.CRITERIONS[cfg.CRITERION.NAME](cfg)
