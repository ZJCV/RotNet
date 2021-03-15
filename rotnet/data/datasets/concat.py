# -*- coding: utf-8 -*-

"""
@date: 2021/3/15 下午7:39
@file: concat.py
@author: zj
@description: 
"""

from torch.utils.data import ConcatDataset

from .base_dataset import BaseDataset
from .fashionmnist import FashionMNIST
from .cifar import CIFAR
from .svhn import SVHN
from zcls.data.datasets.evaluator.general_evaluator import GeneralEvaluator


class Concat(BaseDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        dataset_list = list()
        dataset_list.append(FashionMNIST(root, train=train, transform=transform, target_transform=target_transform,
                                         download=download))
        dataset_list.append(CIFAR(root, train=train, transform=transform, target_transform=target_transform,
                                  download=download, is_cifar100=True))
        dataset_list.append(CIFAR(root, train=train, transform=transform, target_transform=target_transform,
                                  download=download, is_cifar100=False))
        dataset_list.append(SVHN(root, train=train, transform=transform, target_transform=target_transform,
                                 download=download))

        self.data_set = ConcatDataset(dataset_list)
        # 0-359 degrees
        self.classes = [str(i) for i in range(360)]
        self.data_set.classes = self.classes

        self._update_evaluator()

        super(Concat, self).__init__(self.data_set, transform=transform, target_transform=target_transform)

    def _update_evaluator(self):
        self.evaluator = GeneralEvaluator(self.classes, topk=(1, 5))
