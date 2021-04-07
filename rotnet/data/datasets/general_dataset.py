# -*- coding: utf-8 -*-

"""
@date: 2021/4/4 下午2:55
@file: general_dataset.py
@author: zj
@description: 
"""

import torchvision.datasets as datasets

from .base_dataset import BaseDataset


class GeneralDataset(BaseDataset):

    def __init__(self, root, transform=None, target_transform=None, top_k=(1, 5)):
        self.root = root

        data_set = datasets.ImageFolder(root)
        super(GeneralDataset, self).__init__(data_set, transform, target_transform, top_k)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'
