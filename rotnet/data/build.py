# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: trainer.py
@author: zj
@description: 
"""

import torch
from torch.utils.data import DataLoader

from .transforms.build import build_transform
from .datasets.build import build_dataset
from .samplers import IterationBasedBatchSampler


def build_dataloader(cfg, train=True):
    transform, target_transform = build_transform(cfg, train=train)
    dataset_list = cfg.DATASETS.TRAIN if train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list,
                             transform=transform, target_transform=target_transform, is_train=train)

    data_loaders = []

    for dataset in datasets:
        if train:
            # 训练阶段使用随机采样器
            sampler = torch.utils.data.RandomSampler(dataset)
            batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
            batch_size = cfg.DATALOADER.TEST_BATCH_SIZE

        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
        if train:
            batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.TRAIN.MAX_ITER, start_iter=0)

        data_loader = DataLoader(dataset, num_workers=cfg.DATALOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=True)
        data_loaders.append(data_loader)
    if train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
