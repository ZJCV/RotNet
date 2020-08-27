# -*- coding: utf-8 -*-

"""
@date: 2020/8/23 上午9:51
@file: inference.py
@author: zj
@description: 
"""

import os
import numpy as np
from tqdm import tqdm
import torch
import logging

from rotnet.data.build import build_dataloader
from rotnet.util.logger import setup_logger


def compute_on_dataset(model, data_loader, device):
    results_dict = {}
    acc_dict = {}
    total_error_distance = 0
    for i in range(360):
        results_dict[str(i)] = 0
        acc_dict[str(i)] = 0

    for batch in tqdm(data_loader):
        images, targets = batch
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            outputs = model(images.to(device))
            outputs = [o.to(cpu_device) for o in outputs]
        for target, result in zip(targets, outputs):
            idx = str(target.item())
            pred = torch.argmax(result).item()

            results_dict[idx] += 1
            acc_dict[idx] += (pred == target)
            total_error_distance += abs(pred - target)
    return results_dict, acc_dict, total_error_distance


def inference(cfg, model, data_loader, dataset_name, device):
    dataset = data_loader.dataset
    logger = logging.getLogger(cfg.TEST.NAME)
    logger.info("Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
    results_dict, acc_dict, total_error_distance = compute_on_dataset(model, data_loader, device)
    # print(results_dict)
    # print(acc_dict)
    total_num = np.sum(list(results_dict.values()))
    acc_num = np.sum(list(acc_dict.values()))
    logger.info('acc rate: {:.3f}, avg error distance: {:.3f}'.format(
        1.0 * acc_num / total_num, 1.0 * total_error_distance / total_num))

    for handler in logger.handlers:
        logger.removeHandler(handler)


@torch.no_grad()
def do_evaluation(cfg, model, device):
    model.eval()

    data_loaders_val = build_dataloader(cfg, train=False)
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        inference(cfg, model, data_loader, dataset_name, device)
