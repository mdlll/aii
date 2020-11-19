# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
'''
(训练图像, 训练标签), (测试图像, 测试标签)
(x_train, t_train), (x_test, t_test)
(x_train, 训练图像，有60000个图片，也就是60000行，784列的二维数组
t_train，训练标签，一一对应，确定是哪个
x_test,测试图片，10000张,也是784的二维数组
t_test，测试标签，一一对应，确定是哪个
'''
img = x_train[1]
label = t_train[1]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状从一维数组变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
