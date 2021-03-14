# -*- coding: utf-8 -*-

"""
@date: 2021/3/13 下午7:07
@file: transforms.py
@author: zj
@description: 
"""

import numpy as np
import math
import cv2


def rotate(img, degree, borderValue):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    dst_h = int(w * math.fabs(math.sin(math.radians(degree))) + h * math.fabs(math.cos(math.radians(degree))))
    dst_w = int(h * math.fabs(math.sin(math.radians(degree))) + w * math.fabs(math.cos(math.radians(degree))))

    matrix = cv2.getRotationMatrix2D(center, degree, 1)
    matrix[0, 2] += dst_w // 2 - center[0]
    matrix[1, 2] += dst_h // 2 - center[1]
    dst_img = cv2.warpAffine(img, matrix, (dst_w, dst_h), borderValue=borderValue)

    # imshow(img, 'src')
    # imshow(dst_img, 'dst')
    # cv2.waitKey(0)
    return dst_img


class Rotate:

    def __init__(self, random=False):
        """
        :param random: 默认为False，表示使用255作为边界填充值；如果为True，则随机选择填充值
        """
        self.random = random

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

        dims = img.ndim
        if self.random:
            high = 256
            borderValue = (
                np.random.randint(low, high=high), np.random.randint(low, high=high),
                np.random.randint(low, high=high)) if dims > 1 else np.random.randint(low, high=high)
        else:
            borderValue = (255, 255, 255) if dims > 1 else 255
        rotate_img = rotate(img, angle, borderValue=borderValue)

        return rotate_img, angle
