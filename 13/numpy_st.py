#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:17:09 2022

@author: zhaokai
"""
import numpy as np

"""
数组的运算

"""
nd = np.random.randint(0,10,size=(3,5))
nd
nd**3
np.power(nd,3)
nd/2

nd>5
nd==6

nd +=100 # 在原来的基础上加100
nd
# 不支持/=
nd2 = nd.copy()

nd -=3

"""
索引和切片

"""

## 一维数组
arr = np.random.randint(0,100,size=10)
arr 

arr[0]
arr[[0,3,5]]  # 根据位置取相应的数据
arr[0:6]
arr[1:8:2]

## 二维
arr2 = np.random.randint(0,100,size=(3,5))
# 逗号隔开
arr2

arr2[0,2]
arr2[1,[1,3]]

# 第二行和第三行以及第四列和第五列的数据
arr2[[1,2],3:]

arr2[[1,2]][:,[3,4]]

arr2[[0,2]][:,[0,2,4]]

# 修改
arr2[2,2] = 1024

arr2[2,[4,1]] = 999
arr2

## 花式索引
arr = np.random.randint(0,100,size=20)
arr

arr[[1,3,5,7]]

cond = arr>60
arr[cond]

# 取出小于30大于80 的数
cond1 = arr<30
cond2 = arr>80
cond = cond1 | cond2
arr[cond]
arr

arr = np.zeros(shape=(10,10),dtype=np.int8())
arr
arr[[0,-1]] = 1 # 行变换
arr[:,[0,-1]] = 1 # 列负值
arr


# 创建一个0-4的5*5 的矩阵
a = np.zeros(shape=(5,5),dtype=np.int8())
b = np.full(((5,5)), 99)

c = np.arange(0,5)

a[:] = c
a

# 等比数列
np.set_printoptions(suppress=True)
np.logspace(0, 10,base=2,num=11)


# 创建10个数  把最大值替换为-100
np.set_printoptions(suppress=True)
arr = np.random.random(10)
print("原数据:",arr)

v = arr.max()

con =arr== v
arr[con] =-100
print("修改之后的数据:",arr)

# 根据某一列进行排序
arr = np.random.randint(0,100,size=5)
arr
arr.sort()
arr
np.sort(arr)[::-1]
arr

index = np.argsort(arr)
print(index)

# 根据第三列进行排序

arr[index]
nd = np.random.randint(0,30,size=(5,5))
nd
#获取第三列
nd[:,2]
index = nd[:,2].argsort()
nd[index]


nd2 = np.random.randint(-5,15,size=(5,6))
nd2
# 方式一
nd2[[1,3]][:,[1,3,4]]
# 方式二
index = np.ix_([1,3],[1,3,4])
nd2[index]

"""
形状改变
"""
nd2 = np.random.randint(-5,15,size=(5,6))
# display(nd2)
# display(nd2)
nd2.reshape(6,5)
# -1 表示最后算
nd2.reshape(-1,5)

"""
数据的叠加
"""
import numpy as np

arr1 = np.random.randint(0,10,size=(2,4))
arr2 = np.random.randint(-5,5,size=(3,4))
# axis =0 默认行合并
np.concatenate([arr1,arr2])

np.concatenate([arr1,arr2,arr1])

arr1 = np.random.randint(0,10,size=(3,6))
arr2 = np.random.randint(-5,5,size=(3,4))

np.concatenate([arr1,arr2],axis = 1)

"""
数据的拆分
"""
nd2 = np.random.randint(-5,15,size=(6,9))
# display(nd2)
# 行拆分  一个数字 平均分
np.split(nd2,2)
np.split(nd2,3)

# 列表 表示 节点进行拆分
np.split(nd2,[],axis=0)

np.split(nd2,[1,4,5],axis=0)

# axis =1 就表示列拆分
np.split(nd2,[1,4,5],axis=1)

"""
数据的转置
"""
nd2 = np.random.randint(-5,15,size=(6,9))
nd2
nd3= nd2.reshape(9,6)
nd3
nd4 = nd2.T
nd4

# 复杂数据可以这样做
# 转置
np.transpose(nd2,axes=[1,0])











