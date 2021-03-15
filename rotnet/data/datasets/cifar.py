# -*- coding: utf-8 -*-

"""
@date: 2020/11/10 下午5:02
@file: cifar.py
@author: zj
@description: 
"""

from torchvision.datasets import CIFAR10, CIFAR100

from .base_dataset import BaseDataset
from .evaluator.general_evaluator import GeneralEvaluator


class CIFAR(BaseDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, is_cifar100=True):
        if is_cifar100:
            self.data_set = CIFAR100(root, train=train, download=download)
        else:
            self.data_set = CIFAR10(root, train=train, download=download)
        # 0-359 degrees
        self.classes = [str(i) for i in range(360)]
        self.data_set.classes = self.classes

        self._update_evaluator()

        super(CIFAR, self).__init__(self.data_set, transform=transform, target_transform=target_transform)

    def _update_evaluator(self):
        self.evaluator = GeneralEvaluator(self.classes, topk=(1, 5))
