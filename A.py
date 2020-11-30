# -*- coding: utf-8 -*-
# @Time    : 2020年11月2020/11/29日16:30
# @Author  : SoonCj
# @Email   : F_aF_a@163.com
# @File    : A.py
# @Software: PyCharm
"""
说明:
接受并处理图片
"""
import cv2
import tensorflow as tf
def getImage1(s):
    print(s)
    s= eval(repr(s).replace('\\n', '').replace('/', '\\\\'))
    print(s)
    img = cv2.imread(s)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


def getImage(img):
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
    img = tf.keras.utils.normalize(img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


if __name__ == '__main__':
    getImage1("C:\\Users\\So on\\Desktop\\7.jpg")

