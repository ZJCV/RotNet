# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: build.py
@author: zj
@description: 
"""

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .rotate import Rotate
from .mnist import FMNIST


def build_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Grayscale(),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        transforms.RandomErasing()
    ])

    target_transform = Rotate()

    return transform, target_transform


def build_dataset(data_dir, transform=None, target_transform=None):
    train_dataset = FMNIST(data_dir, download=True, train=True, transform=transform, target_transform=target_transform)
    test_dataset = FMNIST(data_dir, download=True, train=False, transform=transform, target_transform=target_transform)

    return {'train': train_dataset, 'test': test_dataset}, {'train': len(train_dataset), 'test': len(test_dataset)}


def build_dataloader(data_sets):
    train_dataloader = DataLoader(data_sets['train'], batch_size=128, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(data_sets['test'], batch_size=128, shuffle=True, num_workers=8)

    return {'train': train_dataloader, 'test': test_dataloader}
