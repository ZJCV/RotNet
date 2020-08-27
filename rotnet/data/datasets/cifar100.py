# -*- coding: utf-8 -*-

"""
@date: 2020/8/27 下午8:51
@file: cifar100.py
@author: zj
@description: 
"""

from PIL import Image
import torchvision.datasets as datasets
from .base_dataset import BaseDataset


class CIFAR100(BaseDataset):

    def __init__(self, data_dir, train=True, transform=None, target_transform=None, download=True):
        cifar100 = datasets.CIFAR100(data_dir, train=train, download=download)

        super(CIFAR100, self).__init__(cifar100, transform, target_transform)
