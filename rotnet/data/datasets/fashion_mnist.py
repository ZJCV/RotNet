# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 上午11:09
@file: fashion_mnist.py
@author: zj
@description: 
"""

import numpy as np
from PIL import Image
import torchvision.datasets as datasets
from .base_dataset import BaseDataset


class FashionMNIST(BaseDataset):
    """
    zalandoresearch/fashion-mnist
    https://github.com/zalandoresearch/fashion-mnist
    """

    def __init__(self, data_dir, train=True, transform=None, target_transform=None, download=True):
        fashionmnist = datasets.FashionMNIST(data_dir, train=train, download=download)

        super(FashionMNIST, self).__init__(fashionmnist, transform, target_transform)
