# -*- coding: utf-8 -*-
# @Time    : 2020年11月2020/11/30日16:40
# @Author  : SoonCj
# @Email   : F_aF_a@163.com
# @File    : mnist_model.py
# @Software: PyCharm
"""
说明:

"""
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from A import getImage1

class nnModel:
    def makeModel(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        # 输入层：748（图片是28*28的）
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 图片变平
        # hidden layer1：128
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # hidden layer2：128
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # 输入层：10
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 提供数据
        model.fit(x_train, y_train, epochs=5)
        val_loss, val_acc = model.evaluate(x_test, y_test)
        print("\nloss=", val_loss, "acc=", val_acc)
        model.save("mnist_dnn_model_test.h5")

    def predict(self,s):
        model = tf.keras.models.load_model("mnist_dnn_model_test.h5")
        img = getImage1(s)
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = x_test[98:99]
        for i in range(0, 27):
            x[0][i] = img[i]
        print(x)
        x = tf.keras.utils.normalize(x, axis=1)
        predictions = model.predict([x])
        return np.argmax(predictions[0])