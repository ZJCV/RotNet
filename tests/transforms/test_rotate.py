# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午12:25
@file: transforms.py
@author: zj
@description: 
"""

import cv2

from rotnet.data.transforms.rotate import Rotate


def show(img, dst, angle):
    print(angle)
    print(dst.shape)
    print(img.shape)

    cv2.imshow('dst', dst)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def test_rotate(model, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    print(img.shape)

    dst, angle = model(img)

    return img, dst, angle


if __name__ == '__main__':
    model = Rotate(random=True)
    # gray img rotate
    # gray_img_path = 'tests/assets/gray.png'
    # img, dst, angle = test_rotate(model, gray_img_path)
    # show(img, dst, angle)
    # color img rotate
    color_img_path = 'tests/assets/color.png'
    img, dst, angle = test_rotate(model, color_img_path)
    show(img, dst, angle)

