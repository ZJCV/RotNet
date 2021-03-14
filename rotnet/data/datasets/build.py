# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:21
@file: trainer.py
@author: zj
@description: 
"""

from .fashionmnist import FashionMNIST


def build_dataset(cfg, transform=None, target_transform=None, is_train=True, download=True):
    dataset_name = cfg.DATASET.NAME
    data_dir = cfg.DATASET.DATA_DIR

    if dataset_name == 'FashionMNIST':
        dataset = FashionMNIST(data_dir, train=is_train, transform=transform, target_transform=target_transform,
                               download=download)
    else:
        raise ValueError(f"the dataset {dataset_name} does not exist")

    return dataset
