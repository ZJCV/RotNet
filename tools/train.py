# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: trainer.py
@author: zj
@description: 
"""

import os
import torch
import argparse

from rotnet.config import cfg
from rotnet.data.build import build_dataloader
from rotnet.model.build import build_model, build_criterion
from rotnet.optim.build import build_optimizer, build_lr_scheduler
from rotnet.engine.trainer import do_train
from rotnet.engine.inference import do_evaluation
from rotnet.util.checkpoint import CheckPointer
from rotnet.util.logger import setup_logger
from rotnet.util.collect_env import collect_env_info


def train(cfg, device):
    logger = setup_logger(cfg.TRAIN.NAME)

    data_loader = build_dataloader(cfg, train=True)
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    arguments = {"iteration": 0}
    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=cfg.OUTPUT.DIR,
                                save_to_disk=True, logger=logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    model = do_train(cfg, arguments,
                     data_loader, model, criterion, optimizer, lr_scheduler, checkpointer,
                     device, logger)
    return model


def main():
    parser = argparse.ArgumentParser(description='RotNet Training With PyTorch')
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=2500, type=int,
                        help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=bool)

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.TRAIN.LOG_STEP = args.log_step
    cfg.TRAIN.SAVE_STEP = args.save_step
    cfg.TRAIN.EVAL_STEP = args.eval_step
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT.DIR):
        os.makedirs(cfg.OUTPUT.DIR)

    logger = setup_logger("RotNet", save_dir=cfg.OUTPUT.DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = train(cfg, device)

    logger.info('Start final evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    do_evaluation(cfg, model, device)


if __name__ == '__main__':
    main()
