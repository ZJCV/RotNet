# -*- coding: utf-8 -*-

"""
@date: 2021/2/23 下午8:22
@file: fashionmnist.py
@author: zj
@description: 
"""

import torchvision.datasets as datasets

from .base_dataset import BaseDataset
from zcls.data.datasets.evaluator.general_evaluator import GeneralEvaluator


class FashionMNIST(BaseDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.data_set = datasets.FashionMNIST(root, train=train, transform=transform, target_transform=target_transform,
                                              download=download)
        # 0-359 degrees
        self.classes = [str(i) for i in range(360)]
        self.data_set.classes = self.classes

        self._update_evaluator()

        super(FashionMNIST, self).__init__(self.data_set, transform=transform, target_transform=target_transform)

    def _update_evaluator(self):
        self.evaluator = GeneralEvaluator(self.classes, topk=(1, 5))
