#!/usr/bin/env python
#  -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os
import shutil


def mkdir_if_not_exist(dir, is_delete=False):
    """
    创建文件夹
    :param dirs: 文件夹列表
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir):
                shutil.rmtree(dir)
                print u'[INFO] 文件夹 "%s" 存在, 删除文件夹.' % dir

        if not os.path.exists(dir):
            os.makedirs(dir)
            print u'[INFO] 文件夹 "%s" 不存在, 创建文件夹.' % dir
        return True
    except Exception as e:
        print '[Exception] %s' % e
        return False
