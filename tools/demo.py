# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午4:20
@file: demo.py
@author: zj
@description: 
"""

import cv2
import glob
import os
import time

import torch
from PIL import Image

import argparse
import numpy as np

from rotnet.config import cfg
from rotnet.model.build import build_model
from rotnet.data.transforms.build import build_transform
from rotnet.util.checkpoint import CheckPointer
from rotnet.util.utils import rotate


@torch.no_grad()
def run_demo(cfg, ckpt, images_dir, output_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg).to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT.DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transform, _ = build_transform(cfg, train=False)

    model.eval()
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = Image.open(image_path)
        images = transform(image).unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        outputs = model(images.to(device))[0].to(cpu_device).numpy()
        pred_angle = np.argmax(outputs)
        inference_time = time.time() - start

        meters = ' | '.join(
            [
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                f'predicted angle: {pred_angle}'
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        res_img = rotate(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), -1 * pred_angle)
        res_img_path = os.path.join(output_dir, f'{image_name}.jpg')
        cv2.imwrite(res_img_path, res_img)


def main():
    parser = argparse.ArgumentParser(description="RotNet Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str,
                        help='Specify a image dir to save demo images.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             images_dir=args.images_dir,
             output_dir=args.output_dir)


if __name__ == '__main__':
    main()
