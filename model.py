# -*- coding: utf-8 -*-
# @Time    : 2020年11月2020/11/29日17:09
# @Author  : SoonCj
# @Email   : F_aF_a@163.com
# @File    : NNmodel.py
# @Software: PyCharm
"""
说明:训练模型
"""
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.python.keras import layers

# 载入训练数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# normalize图片处理  /255
# normalized_x_train = tf.keras.utils.normalize(x_train, axis=1)
normalized_x_train = tf.keras.utils.normalize(x_train)
normalized_x_test = tf.keras.utils.normalize(x_test)
# one hot 标签处理，训练要求的
one_hot_y_train = tf.one_hot(y_train, 10)
one_hot_y_test = tf.one_hot(y_test, 10)

# 创建DNN模型：32神经元*32神经元*10神经元
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 图片变平
# model.add(layers.Dense(128, activation=tf.nn.relu))
model.add(layers.Dense(32, activation=tf.keras.activations.relu, input_shape=(784,)))
# model.add(layers.Dense(128, activation=tf.nn.relu))
model.add(layers.Dense(32, activation=tf.keras.activations.relu))
# model.add(layers.Dense(10, activation=tf.nn.softmax))
model.add(layers.Dense(10, activation=tf.keras.activations.softmax))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='rmsprop',loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
# 训练模型
model.fit(normalized_x_train, y_train, epochs=5)
# 图形显示训练结果，调整epochs，初步避免over fitting
val_loss, val_acc = model.evaluate(normalized_x_test, y_test)
print("loss=", val_loss, "acc=", val_acc)
# 保存模型
model.save("mnist_dnn_model_test.h5")
# 读取模型
model = tf.keras.models.load_model("work/mnist_dnn_model_test.h5")

import cv2
img = cv2.imread("C:\\Users\\So on\\Desktop\\1.jpg")

# 横的图片 中心切成正方形
height = int(img.shape[0])
width = int(img.shape[1])
col_start = (width - height) / 2  # 列开头
col_end = col_start + height  # 列结束
img = img[:, int(col_start):int(col_end), :]

# 调整为灰度图片
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 变成黑白的图像
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 图像颜色翻转
img = cv2.bitwise_not(img)
# 调整为28*28像素
img = cv2.resize(img, (28, 28))
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = img.reshape(-1,784)
prediction = model.predict(img)
