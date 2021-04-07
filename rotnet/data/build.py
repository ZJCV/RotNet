# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: trainer.py
@author: zj
@description: 
"""

from zcls.data.dataloader.build import build_dataloader

from .datasets.build import build_dataset
from .transforms.build import build_transform


def build_data(cfg, is_train=True):
    transform, target_transform = build_transform(cfg, is_train=is_train)
    dataset = build_dataset(cfg, transform=transform, target_transform=target_transform, is_train=is_train)

    return build_dataloader(cfg, dataset, is_train=is_train)
