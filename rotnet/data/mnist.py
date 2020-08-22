# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 上午11:09
@file: mnist.py
@author: zj
@description: 
"""

import numpy as np
from PIL import Image
from torchvision.datasets import FashionMNIST


class FMNIST(FashionMNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        """
        先进行图像旋转，再进行图像预处理
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is rotation angle of the image
        """
        img, target = self.data[index], int(self.targets[index])

        if self.target_transform is not None:
            img, target = self.target_transform(img.numpy())

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        # print(img.shape, target)
        return img, target


if __name__ == '__main__':
    dataset = FMNIST('../../data/', train=True)
    img, target = dataset.__getitem__(10)
    print(np.array(img).shape)
    print(target)
