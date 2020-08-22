# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:30
@file: build.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from torch.nn.modules import Sequential
from torchvision.models import mobilenet_v2


def build_model(num_classes=1000):
    model = mobilenet_v2(pretrained=True)
    # print(model)

    model_list = list(model.features.children())
    # print(model_list[0][0])

    input_layer = model_list[0][0]
    in_features = input_layer.in_channels
    out_features = input_layer.out_channels
    kernel_size = input_layer.kernel_size
    stride = input_layer.stride
    padding = input_layer.padding
    bias = input_layer.bias
    # print(in_features, out_features, kernel_size, stride, padding, bias)

    model_list[0][0] = nn.Conv2d(1, out_features, kernel_size, stride=stride, padding=padding, bias=bias)
    model.features = Sequential(*model_list)

    model_list = list(model.classifier.children())
    in_features = model_list[-1].in_features
    model_list[1] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    model.classifier = Sequential(*model_list)
    return model


def build_criterion():
    return nn.CrossEntropyLoss()


if __name__ == '__main__':
    model = build_model(num_classes=10)
    print(model)

    data = torch.randn((8, 1, 224, 224))
    outputs = model(data)
    print(outputs.shape)
