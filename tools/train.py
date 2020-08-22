# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: build.py
@author: zj
@description: 
"""

import torch

from rotnet.data.build import build_dataset, build_transform, build_dataloader
from rotnet.model.build import build_model, build_criterion
from rotnet.optim.build import build_optimizer, build_lr_scheduler
from rotnet.engine.build import train_model

if __name__ == '__main__':
    epoches = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform, target_transform = build_transform()
    data_dir = './data/'
    data_sets, data_sizes = build_dataset(data_dir, transform, target_transform)
    data_loaders = build_dataloader(data_sets)

    criterion = build_criterion()
    model = build_model(num_classes=360).to(device)
    optimizer = build_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer)

    train_model('MobileNet_v2', model, criterion, optimizer, lr_scheduler, data_loaders, data_sizes,
                epoches=epoches, device=device)
