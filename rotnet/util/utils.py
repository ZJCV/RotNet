# -*- coding: utf-8 -*-

"""
@date: 2020/8/23 下午6:50
@file: utils.py
@author: zj
@description: 
"""

import math
import cv2


def rotate(img, degree, borderValue=(255, 255, 255)):
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
