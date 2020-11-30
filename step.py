# -*- coding: utf-8 -*-
# @Time    : 2020年11月2020/11/29日16:24
# @Author  : SoonCj
# @Email   : F_aF_a@163.com
# @File    : step.py
# @Software: PyCharm
"""
说明:流程步骤
1.手写数字照片
2.切成正方形28*28
3.黑底白字
4.DNN分类模型
5.分类结果
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py


def getImage(s):
    img = cv2.imread(s)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 横的图片 中心切成正方形
    height = img.shape[0]
    width = img.shape[1]
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

    # img = tf.keras.utils.normalize(img)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


class model:

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
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
        # train_result = model.fit(x_train, y_train, epochs=5)
        # plt.plot(train_result.history['accuracy'])
        # plt.plot(train_result.history['val_accuracy'])
        # plt.legend(["Auucracy", "Validation Acc"])
        # plt.show()
        self.model = model
        self.x = x_test[98:99]

    def predict(self, s):
        img = getImage(s)
        x = self.x
        for i in range(0, 27):
            x[0][i] = img[i]
        # print(x)
        predictions = self.model.predict([x])
        return np.argmax(predictions[0])


NNmodel = model()
result = NNmodel.predict("C:\\Users\\So on\\Desktop\\7.jpg")
print(result)
