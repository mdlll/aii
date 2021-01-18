# coding: utf-8
import numpy as np

from ch05.layer_naive import *

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward 前向传播
apple_price = mul_apple_layer.forward(apple, apple_num)  # forward=x.y
price = mul_tax_layer.forward(apple_price, tax)
# forward myself
applePrice = np.dot(apple_num, apple)
totalPrice = np.dot(applePrice, tax)
print(totalPrice)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
# backward myself


print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)
