# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午8:23
@file: mobilenet_v2.py
@author: zj
@description: 
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2

from . import registry


@registry.MODELS.register('mobilenet_v2')
def build_mobilenet_v2(cfg):
    in_features = cfg.MODEL.IN_FEATURES
    num_classes = cfg.MODEL.NUM_CLASSES
    pretrained = cfg.MODEL.PRETRAINED

    model = mobilenet_v2(pretrained=pretrained)
    # 替换输入维度
    model_list = list(model.features.children())
    input_layer = model_list[0][0]
    out_features = input_layer.out_channels
    kernel_size = input_layer.kernel_size
    stride = input_layer.stride
    padding = input_layer.padding
    bias = input_layer.bias

    model_list[0][0] = nn.Conv2d(in_features, out_features, kernel_size, stride=stride, padding=padding, bias=bias)
    model.features = nn.Sequential(*model_list)
    # 替换输出类别
    model_list = list(model.classifier.children())
    in_features = model_list[-1].in_features
    model_list[1] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    model.classifier = nn.Sequential(*model_list)
    return model
