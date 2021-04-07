# -*- coding: utf-8 -*-

"""
@date: 2020/11/10 下午5:02
@file: cifar.py
@author: zj
@description: 
"""

from torchvision.datasets import CIFAR10, CIFAR100

from .base_dataset import BaseDataset


class CIFAR(BaseDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, top_k=(1, 5), is_cifar100=True):
        self.root = root

        dataset = CIFAR100 if is_cifar100 else CIFAR10
        data_set = dataset(root, train=train, download=True)
        super(CIFAR, self).__init__(data_set, transform, target_transform, top_k)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'
