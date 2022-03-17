# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 14:36:55 2022

@author: Administrator
"""

import numpy as np
list1 = [1,2,4,5,6,3,7,8]
v = np.array(list1)
print(v)

# 一维变二维
r1=v.reshape(2,4)
print(r1)
r1.ndim

# 一维变三维
r2 = v.reshape(2,2,2)
print(r2)


# -1 表示自己计算
r3=v.reshape(2,-1)
print(r2)


# 二维变三维 
r4 = r1.reshape(1,2,4,)
print(r4)

# resize 不显示  直接修改原始数组
v.resize(4,2)
print(v)


# 将三维降到二维,但是 转化不到 一维
r5 = v.reshape(2,4)
print(r5)

# 将高维转化为一维
r6 = v.ravel()
print(r6)
r6.ndim

# flatten() 将高维数据转化为低维  摊平
r7 = v.flatten()
print(r7)
r7.ndim

# 小补充
# 也可以强制转化
 
 

