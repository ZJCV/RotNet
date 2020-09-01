# -*- coding: utf-8 -*-

"""
@date: 2020/8/23 上午9:51
@file: inference.py
@author: zj
@description: 
"""

from tqdm import tqdm
import torch
import logging

from rotnet.util.metrics import topk_accuracy
from rotnet.data.build import build_dataloader


def compute_on_dataset(model, data_loader, device):
    cpu_device = torch.device("cpu")
    running_acc = 0.0

    for batch in tqdm(data_loader):
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(images)
            topk_list = topk_accuracy(outputs, targets, topk=(1,))

            running_acc += topk_list[0].to(cpu_device).item()
    return running_acc / len(data_loader)


def inference(cfg, model, data_loader, dataset_name, device):
    dataset = data_loader.dataset
    logger = logging.getLogger(cfg.TEST.NAME)
    logger.info("Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
    mean_acc = compute_on_dataset(model, data_loader, device)
    logger.info('acc rate: {:.3f}'.format(mean_acc))

    for handler in logger.handlers:
        logger.removeHandler(handler)


@torch.no_grad()
def do_evaluation(cfg, model, device):
    model.eval()

    data_loaders_val = build_dataloader(cfg, train=False)
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        inference(cfg, model, data_loader, dataset_name, device)
