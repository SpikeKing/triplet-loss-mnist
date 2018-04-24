#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from time import time

import datetime
import numpy as np

from data_loaders.triplet_dl import TripletDL
from infers.triplet_infer import TripletInfer
from utils.config_utils import process_config, get_test_args


def main_test():
    print '[INFO] 解析配置...'
    parser = None
    config = None

    try:
        args, parser = get_test_args()
        config = process_config(args.config)
    except Exception as e:
        print '[Exception] 配置无效, %s' % e
        if parser:
            parser.print_help()
        print '[Exception] 参考: python main_test.py -c configs/triplet_config.json'
        exit(0)

    print '[INFO] 加载数据...'

    print '[INFO] 预测数据...'
    infer = TripletInfer(config=config)
    infer.default_dist()
    infer.test_dist()

    print '[INFO] 预测完成...'


if __name__ == '__main__':
    main_test()
