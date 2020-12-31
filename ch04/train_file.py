# coding: utf-8
import sys, os

from ch04.two_layer_net import TwoLayerNet

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

'''
4.5.2 的机器学习实现
'''


def cross_entropy_error(y, t):
    # 平均交叉熵函数
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]  # 有几行,也就绑了几个batch
    # print(np.log(y[np.arange(batch_size), t]+ 1e-7))#但是not hot-one形式的这个有问题，先存疑
    # return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size # t的数据非hot-one时候使用这个
    return -np.sum(t * np.log(y + 1e-7)) / batch_size  # t的数据是hot-one时候使用这个


np.set_printoptions(threshold=sys.maxsize)
# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# (训练图像, 训练标签), (测试图像, 测试标签)
# (60000, 784) (60000, 10) (10000, 784) (10000, 10)

y = np.array(
    [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]])
t = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
# print(np.pad(t,((0,0),(0,1)),'constant',constant_values=(0,0))) # 在一个矩阵的前后左右添加行列，可指定数值

# 同样的内容怎么这么慢？
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
learn_num = 0.01
train_num = 1000  # design train times
train_loss = []  # recoder loss change
train_size = x_train.shape[0]
batch_size = 10
for i in range(train_num):
    # 抽取10个数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]  # 10张处理好的图
    t_batch = t_train[batch_mask]  # 10个训练标签
    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    # 按照梯度方向更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learn_num * grad[key]
    # 计算损失函数
    loss = network.loss(x_batch, t_batch)
    train_loss.append(loss)
    print(loss)
print(train_loss)

'''
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []


iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # if i % iter_per_epoch == 0:
    #     train_acc = network.accuracy(x_train, t_train)
    #     test_acc = network.accuracy(x_test, t_test)
    #     train_acc_list.append(train_acc)
    #     test_acc_list.append(test_acc)
    #     print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.ylim(0, 9.0)
plt.xlim(0, 11000)
plt.legend(loc='lower right')
plt.show()
'''
