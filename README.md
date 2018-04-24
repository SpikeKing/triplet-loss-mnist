# Triplet Loss 算法

[Triplet Loss](https://arxiv.org/pdf/1503.03832.pdf)是深度学习中的一种损失函数，用于训练**差异性较小**的样本，如人脸等， Feed数据包括锚（Anchor）示例、正（Positive）示例、负（Negative）示例，通过优化锚示例与正示例的距离**小于**锚示例与负示例的距离，实现样本的相似性计算。

<img src="doc/tl_algorithm.png"  width=auto height="200">

数据集：[MNIST](http://yann.lecun.com/exdb/mnist/)

框架：[DL-Project-Template](https://github.com/SpikeKing/DL-Project-Template)

目标：通过Triplet Loss训练模型，实现手写图像的相似性计算。

[工程](https://github.com/SpikeKing/triplet-loss-mnist)：https://github.com/SpikeKing/triplet-loss-mnist

---

## 模型

Triplet Loss的核心是锚示例、正示例、负示例共享模型，通过模型，将锚示例与正示例聚类，远离负示例。

**Triplet Loss Model**的结构如下：

<img src="doc/tl_model.png"  width=auto height="150">

- 输入：三个输入，即锚示例、正示例、负示例，不同示例的**结构**相同；
- 模型：一个共享模型，支持替换为**任意**网络结构；
- 输出：一个输出，即三个模型输出的拼接。

**Shared Model**选择常用的卷积模型，输出为全连接的128维数据：

<img src="doc/base_model.png"  width=auto height="400">

Triplet Loss **损失函数**的计算公式如下：

<img src="doc/tl_formular.png"  width=auto height="80">

---

## 训练

模型参数：

- batch_size：32
- epochs：2

超参数：

- 边界Margin的值设置为``1``。

训练命令：

```text
python main_train.py -c configs/triplet_config.json
```

训练日志：

```text
Using TensorFlow backend.
[INFO] 解析配置...
[INFO] 加载数据...
[INFO] X_train.shape: (60000, 28, 28, 1), y_train.shape: (60000, 10)
[INFO] X_test.shape: (10000, 28, 28, 1), y_test.shape: (10000, 10)
[INFO] 构造网络...
[INFO] model - 锚shape: (?, 128)
[INFO] model - 正shape: (?, 128)
[INFO] model - 负shape: (?, 128)
[INFO] model - triplet_loss shape: (?, 1)
[INFO] 训练网络...
[INFO] trainer - 类别数: 10
Train on 54200 samples, validate on 8910 samples
2018-04-24 10:59:06.130952: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Epoch 1/2
54200/54200 [==============================] - 84s 2ms/step - loss: 0.6083 - val_loss: 0.0515
Epoch 2/2
54200/54200 [==============================] - 83s 2ms/step - loss: 0.0869 - val_loss: 0.0314
```

算法收敛较好，Loss线性下降：

<img src="doc/loss_timeline.png"  width=auto height="200">

TF Graph：

<img src="doc/graph.png"  width=auto height="400">

---

## 验证

**算法效率**（TPS）: 每秒48163次 (0.0207625 ms/t)

测试命令：

```text
python main_test.py -c configs/triplet_config.json
```

测试日志：

```text
Using TensorFlow backend.
[INFO] 解析配置...
[INFO] 加载数据...
[INFO] 预测数据...
展示数据: (10000, 784)
展示标签: (10000,)
日志目录: /Users/wang/workspace/triplet-loss-mnist/experiments/triplet_mnist/logs/default
2018-04-24 11:02:07.874682: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
[INFO] model - triplet_loss shape: (?, 1)
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
anc_input (InputLayer)          (None, 28, 28, 1)    0                                            
__________________________________________________________________________________________________
pos_input (InputLayer)          (None, 28, 28, 1)    0                                            
__________________________________________________________________________________________________
neg_input (InputLayer)          (None, 28, 28, 1)    0                                            
__________________________________________________________________________________________________
model_1 (Model)                 (None, 128)          112096      anc_input[0][0]                  
                                                                 pos_input[0][0]                  
                                                                 neg_input[0][0]                  
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 384)          0           model_1[1][0]                    
                                                                 model_1[2][0]                    
                                                                 model_1[3][0]                    
==================================================================================================
Total params: 112,096
Trainable params: 112,096
Non-trainable params: 0
__________________________________________________________________________________________________
TPS: 272553.829381 (0.003669 ms)
验证结果结构: (10000, 128)
展示数据: (10000, 128)
展示标签: (10000,)
日志目录: /Users/wang/workspace/triplet-loss-mnist/experiments/triplet_mnist/logs/test
[INFO] 预测完成...
```

MNIST验证集的效果：

``` bash
[INFO] trainer - clz 0
[INFO] trainer - distance - min: -15.4567, max: 1.98611, avg: -6.50481
[INFO] acc: 0.996632996633

[INFO] trainer - clz 1
[INFO] trainer - distance - min: -13.09, max: 3.43779, avg: -6.66867
[INFO] acc: 0.99214365881

[INFO] trainer - clz 2
[INFO] trainer - distance - min: -14.2524, max: 2.49437, avg: -5.60508
[INFO] acc: 0.991021324355

[INFO] trainer - clz 3
[INFO] trainer - distance - min: -16.6555, max: 1.21776, avg: -6.32161
[INFO] acc: 0.995510662177

[INFO] trainer - clz 4
[INFO] trainer - distance - min: -14.193, max: 1.65427, avg: -5.90896
[INFO] acc: 0.991021324355

[INFO] trainer - clz 5
[INFO] trainer - distance - min: -14.1007, max: 2.01843, avg: -6.36086
[INFO] acc: 0.994388327722

[INFO] trainer - clz 6
[INFO] trainer - distance - min: -16.8953, max: 2.84421, avg: -8.43978
[INFO] acc: 0.995510662177

[INFO] trainer - clz 7
[INFO] trainer - distance - min: -16.6177, max: 3.49675, avg: -5.99822
[INFO] acc: 0.989898989899

[INFO] trainer - clz 8
[INFO] trainer - distance - min: -14.937, max: 3.38141, avg: -5.4424
[INFO] acc: 0.979797979798

[INFO] trainer - clz 9
[INFO] trainer - distance - min: -16.9519, max: 2.39112, avg: -5.93581
[INFO] acc: 0.985409652076
```

测试的MNIST分布：

<img src="doc/default.png" width=auto height="400">

输出的Triplet Loss MNIST分布：

<img src="doc/test.png" width=auto height="400">

本例仅仅使用2个Epoch，也没有特殊设置超参，实际效果仍有提升空间。

---

By C. L. Wang @ [美图](http://www.meipai.com/)云事业部
