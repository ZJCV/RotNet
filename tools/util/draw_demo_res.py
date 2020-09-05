# -*- coding: utf-8 -*-

"""
@date: 2020/9/5 下午5:19
@file: draw_demo_res.py
@author: zj
@description: 
绘图，比较检测结果
"""

import cv2
import os
import glob
import matplotlib.pyplot as plt


def draw(src_img_list, rotate_img_list, res_img_list):
    assert len(src_img_list) == len(rotate_img_list) == len(res_img_list)

    f = plt.figure()
    plt.figure(figsize=(5, 8))
    plt.suptitle('src-rotate-res')

    rows = len(src_img_list)
    for i in range(rows):
        img_path = src_img_list[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(rows, 3, 3 * i + 1)
        plt.imshow(img, cmap='gray'), plt.axis('off')

        img_path = rotate_img_list[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(rows, 3, 3 * i + 2)
        plt.imshow(img, cmap='gray'), plt.axis('off')

        img_path = res_img_list[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(rows, 3, 3 * i + 3)
        plt.imshow(img, cmap='gray'), plt.axis('off')

    plt.savefig('gray.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    src_img_dir = './demo/src'
    rotate_img_dir = './demo/rotate'
    res_img_dir = './demo/result'

    rotate_img_list = sorted(glob.glob(os.path.join(rotate_img_dir, '*.jpg')))
    res_img_list = sorted(glob.glob(os.path.join(res_img_dir, '*.jpg')))

    print(rotate_img_list)
    print(res_img_list)

    src_img_list = list()
    for img_path in rotate_img_list:
        idx = os.path.splitext(os.path.split(img_path)[1])[0].split('_')[0]
        src_img_path = os.path.join(src_img_dir, f'{idx}.jpg')
        src_img_list.append(src_img_path)
    print(src_img_list)

    draw(src_img_list, rotate_img_list, res_img_list)
