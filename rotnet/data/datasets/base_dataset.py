# -*- coding: utf-8 -*-

"""
@date: 2020/8/27 下午8:51
@file: base_dataset.py
@author: zj
@description: 
"""

import torch
from PIL import Image
import numpy as np


class BaseDataset:

    def __init__(self, dataset, transform=None, target_transform=None):
        self.data = dataset.data
        self.transform = transform
        self.target_transform = target_transform

        self.length = len(dataset)

    def __getitem__(self, index):
        """
        Firstly, the image is rotated, and then the image is preprocessed
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is rotation angle of the image
        """
        img = self.data[index]

        # Convert to numpy.ndarray
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, torch.Tensor):
            img = img.numpy()
        else:
            pass

        # Perform image rotation
        if self.target_transform is not None:
            img, target = self.target_transform(img)
        else:
            # Assume that the initial rotation angle of all training/test images is 0
            target = 0

        # after rotate, make img become PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length
