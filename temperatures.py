import matplotlib
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# import matplotlib.pyplot as plt
# plt.imshow(x_train[1],cmap=plt.cm.binary)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# 输入层：748（图片是28*28的）
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # 图片变平
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
print("loss=", val_loss, "acc=", val_acc)

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
p =  model.predict(img)
