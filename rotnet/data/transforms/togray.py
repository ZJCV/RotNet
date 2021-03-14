# -*- coding: utf-8 -*-

"""
@date: 2021/3/13 下午7:09
@file: togray.py
@author: zj
@description: 
"""

import numpy as np
import cv2


class ToGray:

    def __call__(self, img, angle=None):
        assert isinstance(img, np.ndarray)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img, angle
