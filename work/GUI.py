# -*- coding: utf-8 -*-
# @Time    : 2020年11月2020/11/30日16:27
# @Author  : SoonCj
# @Email   : F_aF_a@163.com
# @File    : GUI.py
# @Software: PyCharm
"""
说明:

"""
from tkinter import messagebox
import tkinter
from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from A import getImage1
import numpy as np


root = Tk()
w_box = 443
h_box = 500
root.title('数字识别__陈静')
root.geometry('465x300')
e = StringVar()
e_entry = Entry(root, textvariable=e)
e_entry.grid(row=6, column=1, padx=10, pady=5)
img1 = 1
global imgGl
imgGl = Label(root, image=None)
imgGl.place(x=10, y=40)


def resize(w_box, h_box, pil_image):
    w, h = pil_image.size
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


def choose_file():
    img1 = tkinter.filedialog.askopenfilename(title='选择文件')
    imgPath = img1
    print(imgPath)
    e.set(img1)
    imgGl.config(image='')
    load = Image.open(img1)
    pil_image_resized = resize(w_box, h_box, load)
    render = ImageTk.PhotoImage(pil_image_resized)
    imgGl.image = render
    imgGl.config(image=render)


def bindFun():
    model = tf.keras.models.load_model("mnist_dnn_model_test.h5")
    img = getImage1(e.get())
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_test[98:99]
    for i in range(0, 27):
        x[0][i] = img[i]
    print(x)
    x = tf.keras.utils.normalize(x, axis=1)
    predictions = model.predict([x])
    print(np.argmax(predictions[0]))

    messagebox.showinfo("识别结果", np.argmax(predictions[0]))


Button(root, text="选择文件", width=10, command=choose_file).grid(row=6, column=0, sticky=W, padx=10, pady=5)
Button(root, text='识别', width=10, command=bindFun).grid(row=6, column=10, sticky=W, padx=10, pady=5)
Button(root, text='退出', width=10, command=root.quit).grid(row=6, column=12, sticky=W, padx=10, pady=5)
mainloop()
