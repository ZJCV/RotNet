# -*- coding: utf-8 -*-

"""
@date: 2021/3/15 下午7:52
@file: svhn.py
@author: zj
@description: 
"""

import torchvision.datasets as datasets

from .base_dataset import BaseDataset
from zcls.data.datasets.evaluator.general_evaluator import GeneralEvaluator


class SVHN(BaseDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        split_name = 'train' if train else 'test'
        self.data_set = datasets.SVHN(root, split=split_name, download=download)
        # 0-359 degrees
        self.classes = [str(i) for i in range(360)]
        self.data_set.classes = self.classes

        self._update_evaluator()

        super(SVHN, self).__init__(self.data_set, transform=transform, target_transform=target_transform)

    def _update_evaluator(self):
        self.evaluator = GeneralEvaluator(self.classes, topk=(1, 5))
