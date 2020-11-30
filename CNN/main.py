# -*- coding: utf-8 -*-
# @Time    : 2020年11月2020/11/29日19:08
# @Author  : SoonCj
# @Email   : F_aF_a@163.com
# @File    : A.py
# @Software: PyCharm
"""
说明:

"""
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(10, (5 * 5), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (5 * 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="rmsprop", loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

train_result = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
