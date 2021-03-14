# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: build.py
@author: zj
@description: 
"""

import os
import numpy as np
import torch

from zcls.engine.trainer import do_train
from zcls.model.recognizers.build import build_recognizer
from zcls.model.criterions.build import build_criterion
from zcls.optim.optimizers.build import build_optimizer
from zcls.optim.lr_schedulers.build import build_lr_scheduler
from zcls.util import logging
from zcls.util.checkpoint import CheckPointer
from zcls.util.collect_env import collect_env_info
from zcls.util.distributed import init_distributed_training, get_device, get_local_rank, synchronize
from zcls.util.misc import launch_job
from zcls.util.parser import parse_args, load_config

from rotnet.data.build import build_dataloader

logger = logging.get_logger(__name__)


def train(cfg):
    # Set up environment.
    init_distributed_training(cfg)
    local_rank_id = get_local_rank()

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED + 10 * local_rank_id)
    torch.manual_seed(cfg.RNG_SEED + 10 * local_rank_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info('init start')
    # 迭代轮数从１开始计数
    arguments = {"cur_epoch": 1}

    device = get_device(local_rank_id)
    model = build_recognizer(cfg, device)
    criterion = build_criterion(cfg, device)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=cfg.OUTPUT_DIR,
                                save_to_disk=True)
    if cfg.TRAIN.RESUME:
        logger.info('resume start')
        extra_checkpoint_data = checkpointer.load(map_location=device)
        if isinstance(extra_checkpoint_data, dict):
            arguments['cur_epoch'] = extra_checkpoint_data['cur_epoch']
            if cfg.LR_SCHEDULER.IS_WARMUP:
                logger.info('warmup start')
                if lr_scheduler.finished:
                    optimizer.load_state_dict(lr_scheduler.after_scheduler.optimizer.state_dict())
                else:
                    optimizer.load_state_dict(lr_scheduler.optimizer.state_dict())
                lr_scheduler.optimizer = optimizer
                lr_scheduler.after_scheduler.optimizer = optimizer
                logger.info('warmup end')
        logger.info('resume end')

    train_data_loader = build_dataloader(cfg, is_train=True)
    test_data_loader = build_dataloader(cfg, is_train=False)

    logger.info('init end')
    synchronize()
    do_train(cfg, arguments,
             train_data_loader, test_data_loader,
             model, criterion, optimizer, lr_scheduler,
             checkpointer, device)


def main():
    args = parse_args()
    cfg = load_config(args)

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    if args.config_file:
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    launch_job(cfg=cfg, init_method=cfg.INIT_METHOD, func=train)


if __name__ == '__main__':
    main()
