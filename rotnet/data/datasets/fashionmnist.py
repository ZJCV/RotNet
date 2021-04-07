# -*- coding: utf-8 -*-

"""
@date: 2021/2/23 下午8:22
@file: fashionmnist.py
@author: zj
@description: 
"""

import torchvision.datasets as datasets

from .base_dataset import BaseDataset


class FashionMNIST(BaseDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, top_k=(1, 5)):
        self.root = root

        data_set = datasets.FashionMNIST(root, train=train, download=True)
        super(FashionMNIST, self).__init__(data_set, transform, target_transform, top_k)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'
