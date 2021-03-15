# -*- coding: utf-8 -*-

"""
@date: 2021/3/15 下午7:07
@file: parse_checkpoint.py
@author: zj
@description: 
"""

import torch


def parse(src_path, dst_path):
    src_model = torch.load(src_path, map_location=torch.device('cpu'))

    dst_model = src_model['model']

    torch.save(dst_model, dst_path)


if __name__ == '__main__':
    parse('./outputs/mbv3_small_se_hsigmoid_fmnist_224_e100/model_0095.pth', './weights/model.pth')
