# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:21
@file: trainer.py
@author: zj
@description: 
"""

from torch.utils.data import ConcatDataset

from rotnet.config.path_catalog import DatasetCatalog
from .fashion_mnist import FMNIST

_DATASETS = {
    'FashionMNIST': FMNIST,
}


def build_dataset(dataset_list, transform=None, target_transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]

        args['train'] = is_train
        args['transform'] = transform
        args['target_transform'] = target_transform

        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets
    # for training, return a dataset
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
