# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    # W is weight, 通过寻找权重文件的损失函数最大值
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 随机生成2两3列的权重矩阵

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z) # 归一化获得一个假设值
        loss = cross_entropy_error(y, t)  # 交叉熵函数

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t) # 获得一个虚拟函数
dW = numerical_gradient(f, net.W)

'''
f = lambda x:my_test(x)
等价于
def f(x):
	return my_test(x)
'''

print(net.W) # 更加优秀的权重值

print(dW)
