# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午4:20
@file: predict.py
@author: zj
@description: 
"""

import cv2
import torch
from rotnet.data.build import build_test_transform
from rotnet.model.build import build_model
from rotnet.util.checkpoint import CheckPointer

if __name__ == '__main__':
    epoches = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = build_model(num_classes=360).to(device)
    output_dir = './outputs'
    checkpointer = CheckPointer(model, save_dir=output_dir)
    checkpointer.load()

    transform = build_test_transform()
    img_path = 'imgs/RotNet.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    res_img = transform(img).unsqueeze(0)
    print(res_img.shape)

    outputs = model(res_img.to(device))
    _, preds = torch.max(outputs, 1)
    print(preds)
