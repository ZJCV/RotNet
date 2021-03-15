# -*- coding: utf-8 -*-

"""
@date: 2021/3/15 下午4:25
@file: rotnet.py
@author: zj
@description: 
"""

config_file = 'configs/mbv3_small_se_hsigmoid_fmnist_224_e100.yaml'

model_url = 'https://cloud.zhujian.tech:9300/s/rGwWKbCmkGfwxA3'

import torch
from zcls.config import cfg
from zcls.model.recognizers.build import build_recognizer


def rotnet(pretrained=False, **kwargs):
    cfg.merge_from_file(config_file)
    model = build_recognizer(cfg, torch.device('cpu'))
    # if pretrained:
    #     state_dict = torch.load(checkpoint)
    #     model.load_state_dict(state_dict)
    return model
