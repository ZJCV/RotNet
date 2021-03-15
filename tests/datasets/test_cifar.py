# -*- coding: utf-8 -*-

"""
@date: 2021/3/15 下午8:05
@file: test_cifar.py
@author: zj
@description: 
"""

import numpy as np

from rotnet.data.datasets.cifar import CIFAR


def test_cifar10():
    root_data = './data/cifar'
    data_set = CIFAR(root_data, is_cifar100=False)

    print(data_set.classes)
    print(len(data_set))

    img, target = data_set.__getitem__(100)
    print(np.array(img).shape, target)


if __name__ == '__main__':
    test_cifar10()
