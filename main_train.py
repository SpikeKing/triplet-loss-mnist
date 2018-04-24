#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18

参考:
NumPy FutureWarning
https://stackoverflow.com/questions/48340392/futurewarning-conversion-of-the-second-argument-of-issubdtype-from-float-to
"""

from data_loaders.triplet_dl import TripletDL
from infers.triplet_infer import TripletInfer
from models.triplet_model import TripletModel
from trainers.triplet_trainer import TripletTrainer
from utils.config_utils import process_config, get_train_args
import numpy as np


def main_train():
    """
    训练模型

    :return:
    """
    print '[INFO] 解析配置...'

    parser = None
    config = None

    try:
        args, parser = get_train_args()
        config = process_config(args.config)
    except Exception as e:
        print '[Exception] 配置无效, %s' % e
        if parser:
            parser.print_help()
        print '[Exception] 参考: python main_train.py -c configs/triplet_config.json'
        exit(0)
    # config = process_config('configs/triplet_config.json')

    print '[INFO] 加载数据...'
    dl = TripletDL(config=config)

    print '[INFO] 构造网络...'
    model = TripletModel(config=config)

    print '[INFO] 训练网络...'
    trainer = TripletTrainer(
        model=model.model,
        data=[dl.get_train_data(), dl.get_test_data()],
        config=config)
    trainer.train()
    print '[INFO] 训练完成...'


if __name__ == '__main__':
    main_train()
