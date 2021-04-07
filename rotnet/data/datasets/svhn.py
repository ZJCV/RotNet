# -*- coding: utf-8 -*-

"""
@date: 2021/3/15 下午7:52
@file: svhn.py
@author: zj
@description: 
"""

import torchvision.datasets as datasets

from .base_dataset import BaseDataset


class SVHN(BaseDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, top_k=(1, 5)):
        self.root = root

        split_name = 'train' if train else 'test'
        data_set = datasets.SVHN(root, split=split_name, download=True)

        super(SVHN, self).__init__(data_set, transform, target_transform, top_k)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'
