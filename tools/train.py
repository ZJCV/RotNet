# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: build.py
@author: zj
@description: 
"""

import os
import torch

from rotnet.data.build import build_dataset, build_train_transform, build_dataloader
from rotnet.model.build import build_model, build_criterion
from rotnet.optim.build import build_optimizer, build_lr_scheduler
from rotnet.engine.build import train_model
from rotnet.util.checkpoint import CheckPointer
from rotnet.config import cfg

if __name__ == '__main__':
    cfg.merge_from_file('configs/mobilenet_v2_fashion_mnish.yaml')
    cfg.freeze()

    epoches = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform, target_transform = build_train_transform()
    data_dir = './data/'
    data_sets, data_sizes = build_dataset(data_dir, transform, target_transform)
    data_loaders = build_dataloader(data_sets)

    criterion = build_criterion(cfg)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer)

    output_dir = './outputs'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=output_dir,
                                save_to_disk=True, logger=None)

    train_model('MobileNet_v2', model, criterion, optimizer, lr_scheduler, data_loaders, data_sizes, checkpointer,
                epoches=epoches, device=device)
