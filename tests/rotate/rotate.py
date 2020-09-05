# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午12:25
@file: rotate.py
@author: zj
@description: 
"""

import os
import glob
import cv2

from rotnet.data.transforms.rotate import Rotate


def batch_rotate(img_dir, res_dir):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    model = Rotate()
    file_list = glob.glob(os.path.join(img_dir, '*.jpg'))
    for file_path in file_list:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        dst, angle = model(img)

        dst_img_path = os.path.join(res_dir, f'{angle}.jpg')
        cv2.imwrite(dst_img_path, dst)


def test():
    model = Rotate()
    img = cv2.imread('tests/rotate/1.png', cv2.IMREAD_GRAYSCALE)

    print(type(img))
    dst, angle = model(img)

    print(angle)
    print(dst.shape)
    print(img.shape)

    cv2.imshow('dst', dst)
    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    batch_rotate('demo/src', 'demo/rotate')
