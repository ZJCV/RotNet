# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: trainer.py
@author: zj
@description: 
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

from .datasets.build import build_dataset
from .transforms.build import build_transform
import zcls.util.distributed as du


def build_dataloader(cfg, is_train=True):
    transform, target_transform = build_transform(cfg, is_train=is_train)
    dataset = build_dataset(cfg, transform=transform, target_transform=target_transform, is_train=is_train)

    world_size = du.get_world_size()
    num_gpus = cfg.NUM_GPUS
    rank = du.get_rank()
    if is_train:
        batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE
        drop_last = True

        if num_gpus > 1:
            sampler = DistributedSampler(dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=True)
        else:
            sampler = RandomSampler(dataset)
    else:
        batch_size = cfg.DATALOADER.TEST_BATCH_SIZE
        drop_last = False

        if num_gpus > 1:
            sampler = DistributedSampler(dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=False)
        else:
            sampler = SequentialSampler(dataset)

    data_loader = DataLoader(dataset,
                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                             sampler=sampler,
                             batch_size=batch_size,
                             drop_last=drop_last,
                             pin_memory=True)

    return data_loader
