# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

from zcls.data.transforms.build import parse_transform

from .rotate import Rotate


def build_target_transform():
    """
    Rotate image randomly. The test set and training set are rotated randomly
    """
    return Rotate(random=True)


def build_transform(cfg, is_train=True):
    return parse_transform(cfg, is_train), build_target_transform()
