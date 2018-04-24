# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os
import random
import warnings

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_fscore_support

from bases.trainer_base import TrainerBase
from root_dir import ROOT_DIR
from utils.np_utils import prp_2_oh_array
from utils.utils import mkdir_if_not_exist


class TripletTrainer(TrainerBase):
    def __init__(self, model, data, config):
        super(TripletTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        train_dir = os.path.join(ROOT_DIR, self.config.tb_dir, "train")
        mkdir_if_not_exist(train_dir)
        self.callbacks.append(
            TensorBoard(
                log_dir=train_dir,
                write_images=True,
                write_graph=True,
            )
        )

        # self.callbacks.append(FPRMetric())
        # self.callbacks.append(FPRMetricDetail())

    def train(self):
        x_train = self.data[0][0]
        y_train = np.argmax(self.data[0][1], axis=1)
        x_test = self.data[1][0]
        y_test = np.argmax(self.data[1][1], axis=1)

        clz_size = len(np.unique(y_train))
        print "[INFO] trainer - 类别数: %s" % clz_size
        digit_indices = [np.where(y_train == i)[0] for i in range(10)]
        tr_pairs = self.create_pairs(x_train, digit_indices, clz_size)

        digit_indices = [np.where(y_test == i)[0] for i in range(10)]
        te_pairs = self.create_pairs(x_test, digit_indices, clz_size)

        anc_ins = tr_pairs[:, 0]
        pos_ins = tr_pairs[:, 1]
        neg_ins = tr_pairs[:, 2]

        X = {
            'anc_input': anc_ins,
            'pos_input': pos_ins,
            'neg_input': neg_ins
        }

        anc_ins_te = te_pairs[:, 0]
        pos_ins_te = te_pairs[:, 1]
        neg_ins_te = te_pairs[:, 2]

        X_te = {
            'anc_input': anc_ins_te,
            'pos_input': pos_ins_te,
            'neg_input': neg_ins_te
        }

        self.model.fit(
            X, np.ones(len(anc_ins)),
            batch_size=32,
            epochs=2,
            validation_data=[X_te, np.ones(len(anc_ins_te))],
            verbose=1,
            callbacks=self.callbacks)

        self.model.save(os.path.join(self.config.cp_dir, "triplet_loss_model.h5"))  # 存储模型

        y_pred = self.model.predict(X_te)  # 验证模型
        self.show_acc_facets(y_pred, y_pred.shape[0] / clz_size, clz_size)

    @staticmethod
    def show_acc_facets(y_pred, n, clz_size):
        """
        展示模型的准确率
        :param y_pred: 测试结果数据组
        :param n: 数据长度
        :param clz_size: 类别数
        :return: 打印数据
        """
        print "[INFO] trainer - n_clz: %s" % n
        for i in range(clz_size):
            print "[INFO] trainer - clz %s" % i
            final = y_pred[n * i:n * (i + 1), :]
            anchor, positive, negative = final[:, 0:128], final[:, 128:256], final[:, 256:]

            pos_dist = np.sum(np.square(anchor - positive), axis=-1, keepdims=True)
            neg_dist = np.sum(np.square(anchor - negative), axis=-1, keepdims=True)
            basic_loss = pos_dist - neg_dist
            r_count = basic_loss[np.where(basic_loss < 0)].shape[0]
            print "[INFO] trainer - distance - min: %s, max: %s, avg: %s" % (
                np.min(basic_loss), np.max(basic_loss), np.average(basic_loss))
            print "[INFO] acc: %s" % (float(r_count) / float(n))
            print ""

    @staticmethod
    def create_pairs(x, digit_indices, num_classes):
        """
        创建正例和负例的Pairs
        :param x: 数据
        :param digit_indices: 不同类别的索引列表
        :param num_classes: 类别
        :return: Triplet Loss 的 Feed 数据
        """

        pairs = []
        n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1  # 最小类别数
        for d in range(num_classes):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z3 = digit_indices[dn][i]
                pairs += [[x[z1], x[z2], x[z3]]]
        return np.array(pairs)


class FPRMetric(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            val_y, prd_y, average='macro')
        print " — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f" % (f_score, precision, recall)


class FPRMetricDetail(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, support = precision_recall_fscore_support(val_y, prd_y)

        for p, r, f, s in zip(precision, recall, f_score, support):
            print " — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f - ins %s" % (f, p, r, s)
