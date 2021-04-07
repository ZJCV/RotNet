# -*- coding: utf-8 -*-

"""
@date: 2020/8/27 下午8:51
@file: base_dataset.py
@author: zj
@description: 
"""

import cv2
import torch
from PIL import Image
import numpy as np

from zcls.data.datasets.evaluator.general_evaluator import GeneralEvaluator


class BaseDataset:

    def __init__(self, dataset, transform=None, target_transform=None, top_k=(1, 5)):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        # 0-359 degrees
        self.classes = [str(i) for i in range(360)]

        self.length = len(dataset)
        self._update_evaluator(top_k)

    def __getitem__(self, index):
        """
        Firstly, the image is rotated, and then the image is preprocessed
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is rotation angle of the image
        """
        img, _ = self.dataset.__getitem__(index)

        # Convert to numpy.ndarray
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, torch.Tensor):
            img = img.numpy()
        else:
            pass

        # Convert grayscale to color
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Perform image rotation
        if self.target_transform is not None:
            img, target = self.target_transform(img)
        else:
            # Assume that the initial rotation angle of all training/test images is 0
            target = 0

        # after rotate, make img become PIL Image
        img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length

    def _update_evaluator(self, top_k=(1,)):
        self.evaluator = GeneralEvaluator(self.classes, top_k=top_k)
