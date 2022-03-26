# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:15:59 2022

@author: zhaokai
"""
import numpy as np

"""
广播机制
：当两个数组并不相同，我们可以通过扩展数组的方法来实现相加
或者-  这种机制叫做广播机制

"""
arr1 = np.random.randint(0,10,size =(5,3))

arr2 = np.arange(1,4)

# display(arr1,arr2)

# 广播机制 arr2 变身  变成了5份  每一份和每一行相加

arr1 +arr2

arr3 = np.random.randint(0,10,size=(4,5))

arr4 = arr3.mean(axis=1).reshape(4,1)

arr3 - arr4

"""
通用函数


"""
np.pi

np.sin(np.pi/2)
np.cos(np.pi/2).round(5) # 保留几位小数

# 开平方
np.sqrt(1024)
np.power(64,1/3).round(1)

# 求平方
np.square(8)

def fun(x):
    return x**3
fun(3)

# 幂运算
np.power(3,3)

np.log2(16)

x = np.array([1,6,81,9,4,63,5,0])

y = np.array([11,61,8,9,4,93,5,3])

np.maximum(x,y) # 返回两个数组中比较大的数


arr2 = np.random.randint(0,10,size=(2,2))
arr2

np.inner(arr2[0],arr2) # 返回一维数组的内积

# 取整
a = 3.9999
np.ceil(a)

np.floor(a)

# 裁剪clip
# 小于--拔高值  大于--消除掉值
arr1 = np.random.randint(0,30,size=(20))
np.clip(arr1, 10, 20)

"""
where 函数

"""
x = np.array([1,6,81,9,4])

y = np.array([11,61,8,9,45])

cond = np.array([True,False,True,True,False])

# 根据条件进行筛选
np.where(cond,x,y) # True 选择x,False 选择y的值

arr3 = np.random.randint(0,30,size=20)

np.where(arr3<15,arr3,-15)  # 小于15就是自身的数，大15 替换-15

# 改变学生算错的成绩
scores = np.random.randint(-10,150,size=50)
scores

np.where((scores<0)|(scores >100),1024,scores)

np.where((scores<0)|(scores >100),"算错了，丢你老母！",scores)

"""
集合运算

"""
a = np.array([2,4,6,8])
b = np.array([3,4,5,6])

# 交集
np.intersect1d(a,b)

# 并集
np.union1d(a,b)

# 差集
np.setdiff1d(a,b)


"""
线性代数

"""
# 矩阵乘积
a = np.array([[4,2,3],
              [1,3,1]])  # shape (2*3)

b = np.array([[2,7],
              [-5,-7],
                [9,3]])  # shape (3*2)
print("点乘，矩阵运算：",np.dot(a,b))
a@b
a.dot(b)

# 计算逆矩阵  和原来矩阵相乘 得到单位矩阵
from numpy.linalg import inv,det,eig,svd
np.set_printoptions(suppress =True)
a = np.array([[4,2,3],
              [1,3,1],
              [6,2,3]])  # shape (2*3)

b = inv(a)

a.dot(b)



"""
numpy 高级操作
"""
arr =  np.random.randint(0,10,size=(2,3,4,5))
arr

print("求和：")
arr.sum()

arr.sum(axis = -1) # 对最后一维的数据求和

arr.sum(axis = (-1,-2)).shape

#1、给定一个4维矩阵，如何得到最后两维的和？（提示，指定axis进行计算）

import numpy as np

arr = np.random.randint(0,10,size = (2,3,4,5))

# 最后两维的和，指定轴即可
arr.sum(axis = (2,3))

#2、给定数组[1, 2, 3, 4, 5]，如何得到在这个数组的每个元素之间插入3个0后的新数组？

import numpy as np

arr1 = np.arange(1,6)
print(arr1)
arr2 = np.zeros(shape = 17,dtype = np.int16)
print(arr2)

# 有间隔的，取出数据并进行替换
arr2[::4] = arr1
print('替换之后的数据为：\n',arr2)

#3、给定一个二维矩阵（5行4列），如何交换其中两行的元素（提示：任意调整，花式索引）？
import numpy as np

arr = np.random.randint(0,100,size = (5,4))

print('调整前：\n', arr)

arr = arr[[0,2,1,3,4]]
print('调整后：\n', arr)

#4、创建一个100000长度的随机数组，使用两种方法对其求三次方（1、for循环；2、NumPy自带方法），并比较所用时间(指令：%%time，可以计算运行时间)

#方式一：（NumPy默认方法，很快）

%%time
import numpy as np
arr = np.random.randint(0,10,size = 100000)
print('原数据',arr[:10])
arr3 = np.power(arr,3)
print('NumPy默认方法求三次幂：',arr3[:10])


# 方式二：（for循环方法，比较慢）

%%time
import numpy as np
arr = np.random.randint(0,10,size = 100000)
print('原数据：', arr[:10])
result = []
for i in arr:
    result.append(i**3)
print('for循环求三次幂结果：', result[:10])

#5、创建一个5行3列随机矩阵和一个3行2列随机矩阵，求矩阵积


import numpy as np

A = np.random.randint(0,10,size = (5,3))
B = np.random.randint(0,10,size = (3,2))
display(A,B)

print('方式一：\n',np.dot(A,B)) # 调用NumPy函数dot
print('方式二：\n',A.dot(B)) # 调用对象方法
print('方式三：\n',A @ B) # 使用符号计算

#6、矩阵的每一行的元素都减去该行的平均值（注意，平均值计算时指定axis，以及减法操作时形状改变）
import numpy as np

X = np.random.randint(0,10,size = (3,5))
display(X)

print('计算每一行平均值：')
display(X.mean(axis = 1))

print('每一行的元素都减去该行的平均值：')
# 注意，求了平均值，形状改变（变成二维，便于广播）
X - X.mean(axis = 1).reshape(-1,1)

#7、打印出以下函数（要求使用np.zeros创建8*8的矩阵）：
"""

 [[0 1 0 1 0 1 0 1]
  [1 0 1 0 1 0 1 0]
  [0 1 0 1 0 1 0 1]
  [1 0 1 0 1 0 1 0]
  [0 1 0 1 0 1 0 1]
  [1 0 1 0 1 0 1 0]
  [0 1 0 1 0 1 0 1]
  [1 0 1 0 1 0 1 0]]

"""
import numpy as np

arr = np.zeros(shape = (8,8),dtype=np.int16)

print('原数据：')
display(arr)
# 将奇数行，进行修改
arr[::2,1::2] = 1

# 将偶数行，进行修改
arr[1::2,::2] = 1

print('修改后的数据是：')
arr

"""
  8、正则化一个5行5列的随机矩阵（数据统一变成0~1之间的数字，相当于进行缩小）
正则的概念：矩阵A中的每一列减去这一列最小值，除以每一列的最大值减去每一列的最小值（提示：轴axis给合适的参数！！！）
"""
import numpy as np

A = np.random.randint(1,10,size = (5,5))
print('原数据为：')
display(A)

# 根据公式进行计算，注意axis，要正确指定
B = (A - A.min(axis = 0))/(A.max(axis = 0) - A.min(axis = 0))
print('正则化后：')
B

"""9、如何根据两个或多个条件过滤numpy数组。加载鸢尾
花数据，根据第一列小于5.0并且第三列大于1.5作为条件，进行
数据筛选。（提示，需要使用逻辑与运算：&）

"""
import numpy as np

iris = np.loadtxt('./iris.csv',delimiter = ',')

cond1 = iris[:,0] < 5 # 第一列小于5.0
cond2 = iris[:,2] > 1.5 # 第三列大于1.5

cond = cond1 & cond2 # 合并条件

# 获取数据
iris[cond]

"""
10、计算鸢尾花数据**每一行**的softmax得分（exp表示自然底数e的幂运算）
"""

import numpy as np

iris = np.loadtxt('./iris.csv',delimiter = ',')

print('未计算之前：\n', iris[:5])

def softmax(x):
    exp = np.exp(x)
    # 每一行求和，并且进行形状改变（变成二维，可以进行广播）
    result = exp/exp.sum(axis = 1).reshape(-1,1)
    return result.round(3) # 保留3位小数

result = softmax(iris)
print('softmax得分：\n',result[:5])


