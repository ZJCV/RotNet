# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 上午11:14
@file: rotate.py
@author: zj
@description: 
"""

import numpy as np

from rotnet.util.utils import rotate


class Rotate:

    def __init__(self, random=False, borderValue=(255, 255, 255)):
        """
        :param random: 默认为False，表示使用borderValue指定的边界填充值；如果为True，则忽略borderValue，随机选择填充值
        :param borderValue: 边界填充值
        """
        self.random = random
        self.borderValue = borderValue

    def __call__(self, img: np.ndarray, angle=None):
        """
        :param img:
        :param angle: 如果为None，则随机选择[0,360)的旋转角度
        :return:
        """
        assert isinstance(img, np.ndarray)

        low = 0
        high = 360
        if not angle:
            angle = np.random.randint(low, high=high)
        if self.random:
            high = 260
            borderValue = (
                np.random.randint(low, high=high), np.random.randint(low, high=high), np.random.randint(low, high=high))
        else:
            borderValue = self.borderValue
        rotate_img = rotate(img, angle, borderValue=borderValue)

        return rotate_img, angle
