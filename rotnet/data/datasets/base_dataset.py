# -*- coding: utf-8 -*-

"""
@date: 2020/8/27 下午8:51
@file: base_dataset.py
@author: zj
@description: 
"""

import cv2
from PIL import Image
import numpy as np


class BaseDataset:

    def __init__(self, dataset, transform=None, target_transform=None):
        self.data = dataset.data
        self.targets = dataset.targets
        self.transform = transform
        self.target_transform = target_transform

        self.length = len(dataset)

    def __getitem__(self, index):
        """
        先进行图像旋转，再进行图像预处理
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is rotation angle of the image
        """
        img, target = self.data[index], int(self.targets[index])
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.target_transform is not None:
            img, target = self.target_transform(img)
        else:
            # 假定所有训练/测试图像的初始旋转角度为0
            target = 0

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        # print(img.shape, target)
        return img, target

    def __len__(self):
        return self.length
