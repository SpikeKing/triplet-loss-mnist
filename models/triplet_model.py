# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os
from keras import Input, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, K, merge, Concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

from bases.model_base import ModelBase


class TripletModel(ModelBase):
    """
    TripletLoss模型
    """

    MARGIN = 1.0  # 超参

    def __init__(self, config):
        super(TripletModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = self.triplet_loss_model()  # 使用Triplet Loss训练Model

    def triplet_loss_model(self):
        anc_input = Input(shape=(28, 28, 1), name='anc_input')  # anchor
        pos_input = Input(shape=(28, 28, 1), name='pos_input')  # positive
        neg_input = Input(shape=(28, 28, 1), name='neg_input')  # negative

        shared_model = self.base_model()  # 共享模型

        std_out = shared_model(anc_input)
        pos_out = shared_model(pos_input)
        neg_out = shared_model(neg_input)

        print "[INFO] model - 锚shape: %s" % str(std_out.get_shape())
        print "[INFO] model - 正shape: %s" % str(pos_out.get_shape())
        print "[INFO] model - 负shape: %s" % str(neg_out.get_shape())

        output = Concatenate()([std_out, pos_out, neg_out])  # 连接
        model = Model(inputs=[anc_input, pos_input, neg_input], outputs=output)

        plot_model(model, to_file=os.path.join(self.config.img_dir, "triplet_loss_model.png"),
                   show_shapes=True)  # 绘制模型图
        model.compile(loss=self.triplet_loss, optimizer=Adam())

        return model

    @staticmethod
    def triplet_loss(y_true, y_pred):
        """
        Triplet Loss的损失函数
        """

        anc, pos, neg = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:]

        # 欧式距离
        pos_dist = K.sum(K.square(anc - pos), axis=-1, keepdims=True)
        neg_dist = K.sum(K.square(anc - neg), axis=-1, keepdims=True)
        basic_loss = pos_dist - neg_dist + TripletModel.MARGIN

        loss = K.maximum(basic_loss, 0.0)

        print "[INFO] model - triplet_loss shape: %s" % str(loss.shape)
        return loss

    def base_model(self):
        """
        Triplet Loss的基础网络，可以替换其他网络结构
        """
        ins_input = Input(shape=(28, 28, 1))
        x = Conv2D(32, kernel_size=(3, 3), kernel_initializer='random_uniform', activation='relu')(ins_input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, kernel_size=(3, 3), kernel_initializer='random_uniform', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        out = Dense(128, activation='relu')(x)

        model = Model(ins_input, out)
        plot_model(model, to_file=os.path.join(self.config.img_dir, "base_model.png"), show_shapes=True)  # 绘制模型图
        return model
