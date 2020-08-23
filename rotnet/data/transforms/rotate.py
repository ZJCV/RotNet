# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 上午11:14
@file: rotate.py
@author: zj
@description: 
"""

import math
import random
import cv2
import numpy as np

from rotnet.util.utils import rotate


class Rotate:

    def __call__(self, img: np.ndarray):
        assert isinstance(img, np.ndarray)

        angle = random.randint(0, 359)
        rotate_img = rotate(img, angle)

        return rotate_img, angle
