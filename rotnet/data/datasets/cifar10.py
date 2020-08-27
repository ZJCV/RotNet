# -*- coding: utf-8 -*-

"""
@date: 2020/8/27 下午4:56
@file: cifar10.py
@author: zj
@description: 
"""

from PIL import Image
import torchvision.datasets as datasets
from .base_dataset import BaseDataset


class CIFAR10(BaseDataset):

    def __init__(self, data_dir, train=True, transform=None, target_transform=None, download=True):
        cifar10 = datasets.CIFAR10(data_dir, train=train, download=download)

        super(CIFAR10, self).__init__(cifar10, transform, target_transform)
