# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:26:01 2022

@author: Administrator
"""

import numpy as np

#1.arrange() -》range(start,end,step)
r1 = np.arange(0,9,1)
print(r1)

#2.linspace(start,end,nums)左右都是闭区间
r2 = np.linspace(0,1,11)
print(r2)

# 选取某个元素
r1[0]
r1[3]

# 选取某些元素
r1[[1,3,-1]]

# 切片
r1[0:6]

# 修改
r1[0] = 9
print(r1)


# 批量修改
r1[[1,3,-1]]= 0
r1[0:6] = 26

"""
二维数组的操作
"""
array1 = np.arange(1,25).reshape(4,6)
print(array1)

# 选取某个元素
array1[0,0]

# 选取某行元素
array1[1,:]

# 选取某些行
array1[0:2,:]
array1[[0,2],:]

# 选取某列
array1[:,3]

# 选取某些lie
array1[:,0:2]
array1[:,[0,2]]

# 修改元素
array1[1,4]= 100
print(array1)

array1[:,3] = 1000


"""
三维数组的操作
"""
array2 = np.arange(48).reshape(2,4,6)
print(array2)

# 选取某个元素
# 首先确定选取哪一个二维数组
array2[0,0,5]

# 选取某行元素
array2[0,2,:]

# 选取某些行
array2[0,0:2,:]
array2[0,[1,3],:]

# 选取某列
array2[0,:,1]

# 选取某些列
array2[0,:,1:3]
array2[0,:,[1,3]]

# 修改
array2[1,1,2] = 1000
print(array2)

"""
数组的组合
"""
array3 = np.arange(9).reshape(3,3)
array4 = 2*array3
print(array3)
print(array4)

# 水平组合
np.hstack((array3,array4))
np.hstack((array3,array4,array3))

np.concatenate((array3,array4),axis=1)

# 垂直组合
np.vstack((array3,array4))

np.concatenate((array3,array4),axis=0)

"""
数组的切割
"""
array5 = np.arange(1,17).reshape(4,4)
print(array5)

# 水平切割
np.hsplit(array5,2)
np.split(array5,2,axis=1)

# 垂直切割
np.vsplit(array5,2)
np.split(array5,2,axis=0)

# 强制水平切割
np.array_split(array5,3,axis=1)

# 强制垂直切割
np.array_split(array5,3,axis=0)



"""
数组的运算
"""

# 数组的加法[对应位置元素加]
print(array3+array4)
# 数组的减法[对应位置元素减]
print(array3-array4)
# 数组的乘法[对应位置元素相乘]
print(array3*array4)
# 数组的除法[对应位置元素相除]
print(array3/array4)
# 数组的取余[对应位置元素]
print(array3%array4)
print(array4%array3)
# 数组的取整[对应位置元素]
print(array4//array3)



"""
数组的深拷贝  浅拷贝
"""
# 浅拷贝
array6 = np.array([1,2,3])
array7 = array6

# 更改array7d的元素值
array7[0] = 10000
print(array6)
print("+++++++++++++++++++++")
print(array7)


# 深拷贝
array8 = array6.copy()
array8[0] = 999
print(array6)
print("+++++++++++++++++++++")
print(array8)


"""
numpy 里面的随机模块 randint(start,end)
产生一个随机数
(0,10) -->左闭右开的区间
"""  
r3 = np.random.randint(0,11)
print(r3)

# 随机种子

np.random.seed(1000)
r4 = np.random.randint(0,101)
print(r4)

a =[]
for i in range(1000):
    a1 = np.random.randint(1,101)
    a.append(a1)
print(a)

import matplotlib.pyplot as plt
plt.hist(a,color="r")

"""
rand()
主要生成（0，1）中的随机浮点数
np.random.rand()
"""


"""
normal()
生成一些符合正态分布的数值
np.random.normal()
"""
r1 = np.random.normal()
print(r1)


# 生成随机数矩阵
r2 = np.random.randint(0,10,size=(5,5))
print(r2)

r3= np.random.rand(5,5)
print(r3)

r4 = np.random.normal(0,1,size=(5,5))
print(r4)

"""
numpy 里面的一些内置函数
"""
# 1.求方差
r4.var()
# 2.求标准差
r4.std()
# 3.求均值
r4.mean()
# 4.求和
r4.sum()
# 5.求中位数
np.median(r4)
# 6.求矩阵的行的和
r4.sum(axis=1)
# 7.求矩阵的列的和
r4.sum(axis=0)


"""
numpy 里面进行矩阵运算
r2 r3
"""
# 矩阵的运算
# 加减乘除  除（求逆）
#   加减  -->对应元素的加减

# 矩阵的乘法
r2.dot(r3)
np.dot(r2,r3)


# 矩阵的除法  inv()
# 并不是所有矩阵都有逆，即使没有逆，给你一个逆
# 称为（伪逆）
np.linalg.inv(r2)

# 检验求逆算法的正确性
r2.dot(np.linalg.inv(r2))









