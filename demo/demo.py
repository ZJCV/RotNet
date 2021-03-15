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

import argparse
import numpy as np
from PIL import Image

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT

from rotnet.rotnet import rotnet

from rotnet.data.transforms.build import build_transform
from rotnet.data.transforms.rotate import rotate


def parse_args():
    parser = argparse.ArgumentParser(description="RotNet Demo.")
    parser.add_argument(
        "-cfg",
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--images_dir", default='demo/src', type=str,
                        help='Specify a image dir to do prediction.')
    parser.add_argument("--rotate_dir", default='demo/rotate', type=str,
                        help='Specify a image dir to save rotate images.')
    parser.add_argument("--output_dir", default='demo/res', type=str,
                        help='Specify a image dir to save demo images.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args


@torch.no_grad()
def run_demo(cfg, images_dir, rotate_dir, output_dir):
    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    if not os.path.exists(rotate_dir):
        os.makedirs(rotate_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = rotnet(pretrained=True).to(device)
    model.eval()

    cpu_device = torch.device("cpu")
    transform, target_transform = build_transform(cfg, is_train=False)

    for i, image_path in enumerate(image_paths):
        # First rotate the image, then correct it
        t0 = time.time()
        # Input images are converted to gray scale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        t1 = time.time()

        rotate_img, angle = target_transform(image)
        t2 = time.time()

        transform_img = transform(Image.fromarray(rotate_img, mode='L'))
        t3 = time.time()

        outputs = model(transform_img.unsqueeze(0).to(device))[KEY_OUTPUT][0].to(cpu_device).numpy()
        pred_angle = np.argmax(outputs)
        t4 = time.time()

        meters = ' | '.join(
            [
                'load {:03d}ms'.format(round((t1 - t0) * 1000)),
                'rotate {:03d}ms'.format(round((t2 - t1) * 1000)),
                'transform {:03d}ms'.format(round((t3 - t2) * 1000)),
                'inference {:03d}ms'.format(round((t4 - t3) * 1000)),
                f'rotate_angle: {angle}, predicted angle: {pred_angle}'
            ]
        )
        file_name = os.path.basename(image_path)
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), file_name, meters))

        img_name = os.path.splitext(file_name)[0]
        rotate_img_path = os.path.join(rotate_dir, '%s-%d.jpg' % (img_name, angle))
        cv2.imwrite(rotate_img_path, rotate_img)

        res_img = rotate(rotate_img, -1 * pred_angle, 255)
        res_img_path = os.path.join(output_dir, f'{img_name}-{pred_angle}.jpg')
        cv2.imwrite(res_img_path, res_img)


def main():
    args = parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.ckpt is not None:
        cfg.MODEL.RECOGNIZER.PRELOADED = args.ckpt

    cfg.freeze()

    if args.config_file:
        print("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            print(config_str)
    # print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             images_dir=args.images_dir,
             rotate_dir=args.rotate_dir,
             output_dir=args.output_dir)


if __name__ == '__main__':
    main()
