# coding: utf-8
import sys, os
import numpy as np
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
y=np.array([[1,2,6],[3,4,6],[7,6,0]])
print(y)
print(np.argmax(y))