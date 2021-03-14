# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

from zcls.data.transforms.build import parse_transform

from .rotate import Rotate


def build_target_transform(is_train=True):
    """
    Rotate image randomly. The test set and training set are rotated randomly
    """
    if is_train:
        return Rotate(random=True)
    else:
        return Rotate(random=False)


def build_transform(cfg, is_train=True):
    return parse_transform(cfg, is_train), build_target_transform(is_train)
