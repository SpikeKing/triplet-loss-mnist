# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os

import datetime
from keras.datasets import mnist
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import numpy as np

from bases.infer_base import InferBase
from keras.models import load_model

from models.triplet_model import TripletModel
from root_dir import ROOT_DIR
from utils.utils import mkdir_if_not_exist


class TripletInfer(InferBase):
    def __init__(self, name=None, config=None):
        super(TripletInfer, self).__init__(config)
        if name:
            self.model = self.load_model(name)

    def load_model(self, name):
        model = os.path.join(self.config.cp_dir, name)
        return load_model(model)

    def predict(self, data):
        return self.model.predict(data)

    def test_dist(self):
        model_path = os.path.join(self.config.cp_dir, "triplet_loss_model.h5")

        model = load_model(model_path, custom_objects={'triplet_loss': TripletModel.triplet_loss})
        model.summary()

        (_, _), (X_test, y_test) = mnist.load_data()
        anchor = np.reshape(X_test, (-1, 28, 28, 1))
        X = {
            'anc_input': anchor,
            'pos_input': np.zeros(anchor.shape),
            'neg_input': np.zeros(anchor.shape)
        }
        s_time = datetime.datetime.now()
        res = model.predict(X)
        e_time = datetime.datetime.now() - s_time
        micro = float(e_time.microseconds) / float(len(y_test))
        print "TPS: %s (%s ms)" % ((1000000.0 / micro), (micro / 1000.0))
        data = res[:, :128]
        print "验证结果结构: %s" % str(data.shape)
        log_dir = os.path.join(ROOT_DIR, self.config.tb_dir, "test")
        mkdir_if_not_exist(log_dir)
        self.tb_projector(data, y_test, log_dir)

    def default_dist(self):
        (_, _), (X_test, y_test) = mnist.load_data()
        X_test = np.reshape(X_test, (-1, 28 * 28))
        log_dir = os.path.join(ROOT_DIR, self.config.tb_dir, 'default')
        mkdir_if_not_exist(log_dir)
        self.tb_projector(X_test, y_test, log_dir)

    @staticmethod
    def tb_projector(X_test, y_test, log_dir):
        """
        TB的映射器
        :param X_test: 数据
        :param y_test: 标签, 数值型
        :param log_dir: 文件夹
        :return: 写入日志
        """
        print "展示数据: %s" % str(X_test.shape)
        print "展示标签: %s" % str(y_test.shape)
        print "日志目录: %s" % str(log_dir)

        metadata = os.path.join(log_dir, 'metadata.tsv')

        images = tf.Variable(X_test)

        # 把标签写入metadata
        with open(metadata, 'w') as metadata_file:
            for row in y_test:
                metadata_file.write('%d\n' % row)

        with tf.Session() as sess:
            saver = tf.train.Saver([images])  # 把数据存储为矩阵

            sess.run(images.initializer)  # 图像初始化
            saver.save(sess, os.path.join(log_dir, 'images.ckpt'))  # 图像存储于images.ckpt

            config = projector.ProjectorConfig()  # 配置
            # One can add multiple embeddings.
            embedding = config.embeddings.add()  # 嵌入向量添加
            embedding.tensor_name = images.name  # Tensor名称
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = metadata  # Metadata的路径
            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)  # 可视化嵌入向量
