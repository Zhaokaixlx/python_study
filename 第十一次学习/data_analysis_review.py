# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:27:40 2022

@author: zhaokai
"""
import numpy as np
a = np.array([np.nan,  1,2,np.nan,3,4,5])
a[~np.isnan(a)]


b = np.array([1,  2+6j,  5,  3.5+5j])
b[np.iscomplex(b)]


# np.pi:圆周率 3.1415926...
np.pi

# np.e:自然数e 2.718281828459045...
np.e

# 创建数组
a = np.array([2,3,4])
print(a)

a.dtype

b = np.array([1.2, 3.5, 5.1])
b.dtype

c = np.array( [ [1,2], [3,4] ], dtype=complex )
print(c)


s = 'hellow world'
np.array(s)

# 特殊数组类型
np.zeros( (3,4) )

np.ones( (2,3,4), dtype=np.int16 )

np.empty( (2,3) )

# 创建一个全部由2.22组成的数组
np.full((3,4), 2.22)

# 从0到2Π之间，生成100个数
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
x = np.linspace( 0, 2*pi, 100 )
f = np.sin(x)
plt.plot(f)

# 用于创建一个等比数列
a = np.logspace(1, 100, num=50, endpoint=True, base=10.0, dtype=None)
print(a)


"""
数组运算
 对数组做基本的算术运算，将会对整个
 数组的所有元组进行逐一运算，并将运
 算结果保存在一个新的数组内，而不会
 破坏原始的数组。
"""

a = np.array( [20,30,40,50] )
b = np.arange( 4 )
c = a-b
d = a+b
print(c,d)

e = b**2
print(e)

10*np.sin(a)

a<35

A = np.array( [[1,1],
...             [0,1]] )
B = np.array( [[2,0],
...             [3,4]] )
# 元素间相乘
A * B
#矩阵乘法
A @ B
A.dot(B)
B.dot(A)


###对于+=和 *= 这一类操作符，会修改原始的数组，而不是新建一个
a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))
a *= 3
b += a


a = np.random.random((2,3))
# 计算所有元素的总和
a.sum()
#找出最小值
a.min()
# 找出最大值
a.max()

# 对每一列进行求和
b.sum(axis=0)
# 找出每一行的最小值
b.min(axis=1)
# 对每行进行循环累加
b.cumsum(axis=1)

# 使用sort方法对数组或数组某
#一维度进行就地排序，这会修
#改数组本身
a = np.array([[5,6,9],[6,2,9],[1,15,12]])
b=a
b.sort()
print(b)

a.sort(axis=1)
print(a)
b.sort(axis=0)
print(b)



"""
通用函数
 Numpy为我们提供了一些列
 常用的数学函数，比如sin、
 cos、exp等等，这些被称作
 ‘通用函数’（ufunc）。
 在Numpy中，这些函数都是
 对数组的每个元素进行计算，
 并将结果保存在一个新数组中。
abs	逐个元素进行绝对值计算
fabs	复数的绝对值计算
sqrt	平方根
square	平方
exp	自然指数函数
log	e为底的对数
log10	10为底的对数
log2	2为底的对数
sign	计算每个元素的符号值
ceil	计算每个元素的最高整数值
floor	计算每个元素的最小整数值
rint	保留到整数位
modf	分别将元素的小数部分和整数部分按数组形式返回
isnan	判断每个元素是否为NaN，返回布尔值
isfinite	返回数组中的元素是否有限
isinf	返回数组中的元素是否无限
cos	余弦
sin	正弦
tan	余切
arccos	反余弦
arcsin	反正弦
arctan	反余切

下面是部分二元通用函数：

函数名	描述
add	将数组的对应元素相加
subtract	在第二个数组中，将第一个数组中包含的元素去除
multiply	将数组的对应元素相乘
divide	相除
floor_divide	整除，放弃余数
power	幂函数
maxium	逐个元素计算最大值
minimum	逐个元素计算最小值
mod	按元素进行求模运算
"""
B = np.arange(3)
np.exp(B)
np.sqrt(B)

C = np.array([2., -1., 4.])
# add 也就是+的意思
np.add(B, C)


# 分别将元素的小数部分和整数部分按数组形式返回
x = np.array([1.5,1.6,1.7,1.8])
i,j = np.modf(x)
i,j


x = np.array([[1,4],[6,7]])
y = np.array([[2,3],[5,8]])
# 逐个元素计算最大值
np.maximum(x,y)
# 逐个元素计算最小值
np.minimum(x,y)

np.mod(x,y)
np.floor_divide(x,y)


"""
索引切片迭代
"""

### 一维
import numpy as np
a = np.arange(10)**3

a[2]

a[2:5]

# 首先按步长区间切片，然后将每个元素设置为-1000
a[:6:2] = -1000
print(a)
# 反转a

print(a[ : : -1] )

for i in a:
    print(i)

###  二维
b=np.array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])

 # 第3行第4列的元素，注意索引从0开始计数
b[2,3]
b[2][3]

# 第二列中的每一行，注意区间的左闭合右开的特性
b[:5,1]

b[ : ,1]

# 每一列的，第二行和第三行的元素
b[1:3, : ]
#当给与的参数少于轴数时，其它的轴被认为是全选，比如这里获得最后一行，等同于b[-1,:]
b[-1]

for row in b:
       print(row)

# 如果想对多维数组进行类似Python列表
#的那样迭代，可以使用数组的flat属性
for element in b.flat:
        print(element)


"""
添加删除去重
append:将值添加到数组末尾
insert: 沿指定轴将值插入到指定下标之前
delete: 返回删掉某个轴的子数组的新数组
unique: 寻找数组内的唯一元素

"""
a = np.array([[1,2,3],[4,5,6]])

# 附加后，变成了一维的
np.append(a, [7,8,9])
# 附加后，变成了2维的
np.append(a, [[7,8,9]],axis = 0)

# ndarray没有这个方法
a.append([10,11,12])

a = np.array([[1,2],[3,4],[5,6]])


# 在3号位置前插入，变成一维了
np.insert(a,3,[11,12])

# 按行插入
np.insert(a,1,[11],axis = 0)
#按列插入
np.insert(a,1,[11],axis = 1)


a = np.arange(1,13).reshape(3,4)
# 删除指定位置的元素后，变成一维了
np.delete(a,5)
# 并不会修改原来的数组
print(a)

# 删除指定列
np.delete(a,1,axis = 1)

### unique是numpy中非常重要的方法

a = np.array([0,1,4,7,2,1,4,3])
np.unique(a)

b = np.array([[0,1,4,],[7,2,1],[4,3,0]])
np.unique(b)

np.unique(b,axis=0)
np.unique(b,axis=1)

b = np.array([[0,1,4,],[7,2,1],[4,3,0],[0,1,4,]])
np.unique(b,axis=0)

"""
形状变换
可以通过数组的shape属性，查看它的形状

"""
# floor函数取整
a = np.floor(10*np.random.random((3,4)))
a.shape

# 平铺数组成为一维数组
a.ravel()

# 调整形状
a.reshape(6,2)

# 返回转置数组
a.T

# eshape方法不会修改数组本身，resize则正好相反
a
a.resize((2,6))

np.resize(a, (2,7))

# 如果reshape方法的一个参数是-1，那么这个参数的实际值会自动计算得出
a.reshape(3,-1)

"""
堆积数组

"""
a = np.floor(10*np.random.random((2,2)))
b = np.floor(10*np.random.random((2,2)))

np.vstack((a,b))
np.hstack((a,b))

# 引入一个新轴
from numpy import newaxis
np.column_stack((a,b))

a = np.array([4.,2.])
b = np.array([3.,8.])
# 返回一个二维数组
np.column_stack((a,b))

 # 与上面的结果是不一样的
np.hstack((a,b))

# 为a添加一个轴
a[:,newaxis]

np.column_stack((a[:,newaxis],b[:,newaxis]))
np.hstack((a[:,newaxis],b[:,newaxis]))

"""
分割数组

显然有vsplit方法，对数组进行垂直方向的分割；也有array_split方法综合了两个方法的功能，可以指定分割的轴。

"""
a = np.floor(10*np.random.random((2,12)))

# 将数据均匀分割成3份
np.hsplit(a,3)

# 在指定的列位置，分割数组
np.hsplit(a,(3,4))


"""
视图和复制

"""
# 1. 完全不复制
a = np.arange(12)
# 不会创建新的数组对象，而是多了一个引用
b = a
# a和b只是同一个数组的两个名字
b is a

# 会同时修改a的形状
b.shape = 3,4
a.shape

# 2. view视图
# 理解view，只需要注意两者共享数据，此外其它所有都是独立的
# 现在开始c是一个新的数组，并且和a共享数据
c = a.view()
c is a

# c是a数组的数据视图
c.base is a

# 可以看到c没有自己的数据
c.flags.owndata

 # a的形状不会发生改变
c.shape = 2,6
a.shape
# 但是a的数据会跟着发生改变
c[1,0] = 4321
c[0,1] = 999
a

s = a[ : , 1:3]
# s[:]是s的一个视图。注意与s=10区分开，不要混淆。
s[:] = 10
a

### 3. 深度拷贝
# copy方法生成数组的一个完整的拷贝，包括其数据。
d = a.copy()
d[0,0] = 9999
a
d


"""
广播机制
广播允许通用函数以有意义的方式处理异构的输入。也就是让形状不一样的数组在进行运算的时候，能够得到合理的结果。其规则如下：

1.如果所有输入数组的维数都不相同，则会重复在较小数组的形状前面加上“1”，直到所有数组的维数都相同。
2.确保沿特定维度的大小为1的数组的大小与沿该维度的最大形状的数组的大小相同。假定数组元素的值在“广播”数组的该维度上相同。
"""
a=np.array([[0,0,0], [10,10,10],[20,20,20],[30]*3])
b = np.array([0,1,2])
a + b

c= np.array([[1],[2],[3],[4]])
a+c


"""
花式索引
numpy提供了比常规的python序列更多的
索引工具。正如我们前面看到的，除了
按整数和切片索引之外，还可以使用数
组进行索引
"""
a = np.arange(12)**2
#一个包含索引数据的数组
i = np.array( [ 1,1,3,8,5 ] )
a[i]

#一个二维索引数组
j = np.array( [ [ 3, 4], [ 9, 7 ] ] )
# 最终结果和j的形状保持一致
a[j]

### 当被索引的数组是多维数组时，
#将按照它的第一轴进行索引的

palette = np.array( [ [0,0,0],
...                       [255,0,0],
...                       [0,255,0],
...                       [0,0,255],
...                       [255,255,255] ] )

image = np.array( [ [ 0, 1, 2, 0 ],
...                     [ 0, 3, 4, 0 ]  ] )


palette[image]

"""
下面的例子，其实就是从i中拿一个数，
再从j的相同位置拿一个数，组成一个
索引坐标，再去a中找元素。这有个前提
，就是i和j必须是同构的
"""
a = np.arange(12).reshape(3,4)
i = np.array( [ [0,1],[1,2] ] )
j = np.array( [ [2,1],[3,3] ] )
a[i,j]

# 用一个列表作为索引参数
a = np.arange(5)
a[[1,3,4]] = 0

# 也可以用累加函数
a = np.arange(5)
a[[0,1,2]]+=5
print(a)




"""
布尔索引
使用布尔数组进行索引，其实就是我们
显式地选择数组中需要哪些项，不需要
哪些项。

"""
a = np.arange(12).reshape(3,4)
# 通过比较运算，b变成了一个由布尔值组成的数组
b = a > 4

# 生成一个由True值对应出来的一维数组
a[b]

#所有a中大于4的元素被重新赋值为0
a[b] = 0

a[a>4] = 0

"""
使用~可以对布尔值取反，
|表示或，&表示与
"""
a[~b]

a[(a<4)|(a>7)]

(a>3)&(a<8)

a[(a>3)&(a<8)]


"""
统计方法
方法	说明
sum	求和
mean	算术平均数
std	标准差
var	方差
min	最小值
max	最大值
argmax	最大元素在指定轴上的索引
argmin	最小元素在指定轴上的索引
cumsum	累积的和
cumprod	累积的乘积

"""
import numpy as np
a = np.arange(12).reshape(3,4)

a.sum()
a.mean(axis=1)
a.mean(axis =0)
a.std()
a.var()
a.max()

a.cumsum()

a.cumprod()

a.cumprod(axis = 1)

"""
除了以上的统计方法，还有针对布尔数组的三个重要方法：
sum、any和all：
sum : 统计数组或数组某一维度中的True的个数
any： 统计数组或数组某一维度中是否存在一个/多个True，只要有则返回True，否则返回False
all：统计数组或数组某一维度中是否都是True，都是则返回True，否则返回False
"""
a.any()

a.all()



"""
随机数
numpy.random中的部分函数:
 函数	功能
random	返回一个区间[0.0, 1.0)中的随机浮点数
seed	向随机数生成器传递随机状态种子
permutation	返回一个序列的随机排列，或者返回一个乱序的整数范围序列
shuffle	随机排列一个序列
rand	从均匀分布中抽取样本
randint	根据给定的由低到高的范围抽取随机整数
randn	从均值0，方差1的正态分布中抽取样本
binomial	从二项式分布中抽取样本
normal	从正态分布中抽取样本
beta	从beta分布中抽取样本
chisquare	从卡方分布中抽取样本
gamma	从伽马分布中抽取样本
uniform	从均匀[0,1)中抽取样本
"""
import numpy.random as npr
npr.random()
# 生成5个
npr.random(5)
# 生成2行3列
npr.random((2,3))

# 设置种子
npr.seed(42)
npr.randn()

# 生成一个1到10之间的整数
npr.randint(1,10)

# 生成5个1到10之间的整数
npr.randint(1,10,5)

# 指定正态分布的两个重要参数
npr.normal(3,4)

# 生成2行3列
npr.normal(3,4,(2,3))
















"""
Pandas :
    Pandas的主要特点：

快速高效的DataFrame对象，具有默认和自定义的索引。
将数据从不同文件格式加载到内存中的数据对象的工具。
丢失数据的数据对齐和综合处理。
重组和摆动日期集。
基于标签的切片，索引和大数据集的子集。
可以删除或插入来自数据结构的列。
按数据分组进行聚合和转换。
高性能合并和数据加入。
时间序列功能。

"""

"""
Series:Pandas的核心是三大数据结构：Series、DataFrame和Index。绝大多数操作都是围绕这三种结构进行的。
Series是一个一维的数组对象，它包含一个值序列和一个对应的索引序列
"""
import pandas as pd
s = pd.Series([7,-3,4,-2])
print(s)

s.dtype

s.values

s.index

# 可以在创建Series对象的时候指定索引
s2 = pd.Series([7,-3,4,-2], index=['d','b','a','c'])

pd.Series(5,index = list("abcde"))

# 通过index筛选结果
pd.Series({2:'a',1:'b',3:'c'}, index=[3,2])


## 也可以在后期，直接修改index：
s
s.index = ["a","b","c","d"]
print(s)

# 类似Python的列表和Numpy的数组，Series也可以通过索引获取对应的值
s2['a']
s2[['c','a','d']]

# 也可以对Seires执行一些类似Numpy的通用函数操作：
s2[s2>0]
s2*2

import numpy as np
np.exp(s2)

'b' in s2

'e'in s2

### 使用Python的字典来创建Series
dic = {'beijing':35000,'shanghai':71000,'guangzhou':16000,'shenzhen':5000}
s3=pd.Series(dic)

# 自然，具有类似字典的方法
s3.keys()

s3.items()

list(s3.items())

# 添加一个新元素
s3["changsha"] = 20300


city = ['nanjing', 'shanghai','guangzhou','beijing']
s4=pd.Series(dic, index=city)
print(s4)


### 在Pandas中，可以使用isnull和notnull函数来检查缺失的数据
pd.isnull(s4)

pd.notnull(s4)

s4.isnull()

# 可以为Series对象和其索引设置name属性，这有助于标记识别：
s4.name = 'people'
s4.index.name= 'city'
s4


"""
DataFrame
DataFrame是Pandas的核心数据结构，
表示的是二维的矩阵数据表，类似关系
型数据库的结构，每一列可以是不同的
值类型，比如数值、字符串、布尔值等等。
DataFrame既有行索引，也有列索引，它
可以被看做为一个共享相同索引的Series
的字典。


"""
data = {'state':['beijing','beijing','beijing','shanghai','shanghai','shanghai'],'year':[2000,2001,2002,2001,2002,2003],'pop':[1.5, 1.7,3.6,2.4,2.9,3.2]}
f = pd.DataFrame(data)

f.columns

f.index

f.dtypes

# 按行查看
f.values

# head方法查看DataFrame对象的前5行，
# 用tail方法查看后5行。或者head(3)，tail(3)指定查看行数
f.head()

f.tail()

pd.DataFrame(data, columns=['year','state','pop'])

f2 = pd.DataFrame(data, columns=['year','state','pop'],index=['a','b','c','d','e','f'])


# 属性的形式来检索。这种方法bug多，比如属性名不是纯字符串，或者与其它方法同名
f2["year"]
f2["pop"]


# 检索一行却不能通过f2['a']这种方式，而是需要通过loc方法进行选取
f2.loc['a']

# 追加列
f2["debt"] = 12
f2['debt'] = np.arange(1,7)

val = pd.Series([1,2,3],index = ['c','d','f'])
f2['debt'] = val

# 追加行
df = pd.DataFrame(data,index=list('abcdef'))
df1 = df.loc['a']
df.append(df1)


# del方法删除指定的列
f2["new"] = f2.state=="beijing"
del f2['new']

f2.columns


## 类似Numpy的T属性，将DataFrame进行转置
f2.T
f2.index.name = 'order';f2.columns.name='key'
f2
f2.values

"""
DataFrame有一个Series所不具备的方
法，那就是info！通过这个方法，可以看到DataFrame的一些整体信息情况
"""
f.info()









"""
Index

"""
obj = pd.Series(range(3),index = ['a','b','c'])

index = obj.index

index[1:]

index[1] = 'f'  # TypeError

index.size
index.shape
index.ndim
index.dtype

'c' in f2.index
'pop' in f2.columns

# pandas的索引对象可以包含重复的标签
dup_lables = pd.Index(['foo','foo','bar','bar'])

f2.columns = ['year']*4
f2

# 可以使用这个属性来判断是否存在重复的索引
f2.index.is_unique
f2.columns.is_unique


"""
重建索引
reindex方法用于重新为Pandas对象设
置新索引。这不是就地修改，而是会参
照原有数据，调整顺序。
"""
obj=pd.Series([4.5,7.2,-5.3,3.6],index = ['d','b','a','c'])
obj2 = obj.reindex(list('abcde'))

#也可以为缺失值指定填充方式method参数，比如ffill表示向前填充，bfill表示向后填充
obj3 = pd.Series(['blue','purple','yellow'],index = [0,2,4])
obj3.reindex(range(6),method='ffill')




"""
轴向上删除条目
通过drop方法，可以删除Series的一个元
素，或者DataFrame的一行或一列。默认
情况下，drop方法按行删除，且不会修改
原数据，但指定axis=1则按列删除，指定
inplace=True则修改原数据。

"""
import pandas as pd
import numpy as np
s = pd.Series(np.arange(5),index = list("abcde"))

new_s = s.drop('c')

df = pd.DataFrame(np.arange(16).reshape(4,4),columns=['one','two','three','four'])

df.drop(2)
# 指定删除列，而不是默认的行
df.drop('two',axis = 1)

#修改原数据
df.drop(2,inplace=True)



"""
索引和切片
  Series的打印效果，让我们感觉它像
  个二维表格，实际上它还是一维的，
  其索引和numpy的一维数组比较类似，
  但还是有点区别的
"""

se = pd.Series(np.linspace(1,4,5),index=list('abcde'))

se
# 利用我们专门指定的索引进行检索
se['b']

# 实际上默认还有一个从0开始的索引供我们使用# 实际上默认还有一个从0开始的索引供我们使用
se[2]

se[2:4]

# 根据索引顺序，值进行相应的排序，而不是我们认为的按原来的顺序
se[['b','a','d']]

# 左闭右开 ，千万不要写成se[1,3]
se[[1,3]]

se[se>2]

# 什么！居然是左闭右也闭！
se['b':'c']

# 这样会修改原Series
se['b':'c'] = 6


### 如果你的Series是显式的整数
#索引，那么s[1]这样的取值操作会
#使用显式索引，而s[1:3]这样的切片操作却会使用隐式索引。

s = pd.Series(['a','b','c'], index=[1,3,5])

s[1]

s[1:3]
"""
Pandas考虑到了这一点，提供了类似
numpy的行+列的索引标签，也就是
loc和iloc。这两者差一个字母i。
后者是以隐含的整数索引值来索引的，
前者则使用你指定的显式的索引来定位值
。
"""
df = pd.DataFrame(np.arange(16).reshape(4,4),
index=list('abcd'),columns=['one','two','three','four'])

# 使用显式索引值，用逗号分隔行和列参数
df.loc["b",["two","three"]]
# 切片方式，注意区间
df.loc['b':, 'two':'four']

# 用隐含的整数索引检索，但是这个打印格式好别扭
df.iloc[2, [3, 0, 1]]

df.iloc[2]

# 先切片，再布尔判断
df.iloc[:,:3][df.three>5]


"""
算术和广播
  当对两个Series或者DataFrame对象进
  行算术运算的时候，返回的结果是两个
  对象的并集。如果存在某个索引不匹配
  时，将以缺失值NaN的方式体现，并对
  以后的操作产生影响。这类似数据库的
  外连接操作
  add：加法
    sub：减法
    div：除法
    floordiv：整除
    mul：乘法
    pow：幂次方
"""
s1 = pd.Series([4.2,2.6, 5.4, -1.9], index=list('acde'))
s2 = pd.Series([-2.3, 1.2, 5.6, 7.2, 3.4], index= list('acefg'))

s1+s2

s1-s2

s1* s2


df1 = pd.DataFrame(np.arange(9).reshape(3,3),columns=list('bcd'),index=['one','two','three'])
df2 = pd.DataFrame(np.arange(12).reshape(4,3),columns=list('bde'),index=['two','three','five','six'])

df1 + df2

## 防止NaN对后续的影响，很多时候我们
#要使用一些填充值
df1.add(df2,fill_value = 0)

# 也可以这么干
df1.reindex(columns=df2.columns, fill_value=0)

a = np.arange(12).reshape(3,4)
# 取a的第一行，这是一个一维数组
a[0]
# 二维数组减一维数组，在行方向上进行了广播
a - a[0]

# array([[0, 0, 0, 0],
#        [4, 4, 4, 4],
#        [8, 8, 8, 8]])

df = pd.DataFrame(np.arange(12).reshape(4,3),columns=list('bde'),index=['one','two','three','four'])
# 取df的第一行生成一个Series
s = df.iloc[0]

df - s

df + s

s2 = pd.Series(range(3), index=list('bef'))
# 如果存在不匹配的列索引，则引入缺失值
df + s2

# 取df的一列
s3 = df["d"]

df.sub(s3,axis = "index")

"""
函数和映射
  一些Numpy的通用函数对Pandas对象
  也有效：


"""
df = pd.DataFrame(np.random.randn(4,3), columns=list('bde'),index = ['one','two','three','four'])
np.abs(df)

f = lambda x:x.max() - x.min()

df.apply(f)

# 可以指定按行应用f
df.apply(f, axis="columns")

"""
还有更细粒度的apply方法，也就是DataFrame的applymap以及Series的map。它们逐一对每个元素进行操作，而不是整行整列的操作。
"""
f3= lambda x: "%.2f" % x
df.applymap(f3)


# 获取d列，这是一个Series
df["d"].map(f3)



"""
排序和排名
   排序分两种：根据索引排序和根据元素值排序
   索引排序使用的是sort_index方法
"""
import pandas as pd
import numpy as np
s= pd.Series(range(4),index = list("dabc"))
# 根据索引的字母序列排序
s.sort_index()
# 根据索引的值排序
s.sort_values()

df = pd.DataFrame(np.random.randint(10,size=(4,3)), columns=list('edb'),index = ['two','one','five','four'])

df.sort_index()
# 指定按列排序
df.sort_index(axis=1)
 # 默认升序，可以指定为倒序
df.sort_index(axis=1,ascending=False)  # 默认升序，可以指定为倒序

df.sort_values()

s2 = pd.Series([4, np.nan,7,np.nan,-3,2])

s2.sort_values()

df2 = pd.DataFrame({'b':[4,7,-3,2], 'a':[0,1,0,1]})
# 根据某一列里的元素值进行排序
df2.sort_values(by="b")

# 根据某些列进行排序
df2.sort_values(by=['a','b'])

#### 排名   最小的数排为1.0
s = pd.Series([7,-5,7,4,2,0,4])
s.rank()

# 根据观察顺序进行排名位次分配
s.rank(method='first')

# 按最大值排名，并降序排列
s.rank(method='max',ascending=False)

"""
DataFrame则可以根据行或列计算
排名
   average:默认方式，计算平均排名
    min：最小排名
    max：最大排名
    first：观察顺序排名
    dense：类似min，但组间排名总是增加1
"""
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randint(-10,10,(4,3)),columns=list('abc'))

df.rank(axis='columns')



"""
统计和汇总
    方法	描述
    min	最小值
    max	最大值
    idxmin	返回某行或某列最小值的索引
    idxmax	最大值的索引
    cumsum	累加
    cumprod	累乘
    count	统计非NaN的个数
    describe	汇总统计集合
    quantile	计算样本的从0到1间的分位数
    sum	求和
    mean	平均值
    median	中位数（50%）
    mad	平均值的平均绝对偏差
    prod	所有值的积
    var	方差
    std	标准差
    skew	样本偏度，第三时刻值
    kurt	样本峰度，第四时刻值
    diff	计算第一个算术差值
    pct_change	计算百分比
"""
df = pd.DataFrame([[1.4, np.nan],[7.1,-4.2],[np.nan,np.nan],[0.75,-1.1]],index=list('abcd'),columns=['one','two'])

# 默认对列进行求和，并返回一个Series对象，缺失值默认被忽略
df.sum()
# 指定对行进行求和
df.sum(axis='columns')
# 对行求平均值，但不忽略缺失值
df.mean(axis='columns', skipna=False)
# 对行求平均值，忽略缺失值
df.mean(axis='columns', skipna=True)


df.idxmax()
df.idxmin()

df.cumsum()

df.cumprod()

df.count()

### 最重要的describe方法
df.describe()

s=pd.Series(list('aabc'*4))
print(s)
# 对于非数值型，统计类型不一样
s.describe()

"""
还有几个非常重要的方法:

unique：计算唯一值数组，并按观察顺序返回
value_counts:计数，并按多到少排序
isin：判断是否包含成员
"""
s = pd.Series(list('cadaabbcc'))
# 获取去重后的值
uniques = s.unique()
# 计数，默认从多到少排序
s.value_counts()

# 也可以这么调用，并且不排序
s.value_counts(s,sort= False)

# 判断Series里面有没有b和c
mask = s.isin(["b","c"])

s[mask]





"""
文件读取
读取文本文件或磁盘上的其它高效文件格式
与数据库交互
与网络资源，比如Web API进行交互

"""

"""
1. 文本格式数据的读写
   最多的是将表格型的数据读取为
   DataFrame对象。实现这一功能的
   函数有很多，最常用的是read_csv
   和read_table
   函数	说明
        d_csv	读取默认以逗号作为分隔符的文件
        read_table	读取默认以制表符分隔的文件
        read_fwf	从特定宽度格式的文件中读取数据（无分隔符）
        read_clipboard	read_table的剪贴板版本
        read_excel	从EXCEL的XLS或者XLSX文件中读取数据
        read_hdf	读取用pandas存储的HDF5文件
        read_html	从HTML文件中读取所有表格数据
        read_json	从JSON字符串中读取数据
        read_msgpack	读取MessagePack格式存储的任意对象
        read_pickle	读取以Python Pickle格式存储的对象
        read_sas	读取SAS系统中定制存储格式的数据集
        read_sql	将SQL查询的结果读取出来
        read_stata	读取stata格式的数据集
        read_feather	读取Feather二进制格式
"""
import pandas as pd
df = pd.read_excel(r"D:\pycharmcode\第十一次学习\data2.xlsx")

# 使用默认列名

df = pd.read_excel(r"D:\pycharmcode\第十一次学习\data2.xlsx",header=None)


"""
# csv_mindex.csv

key1,key2,value1,value2
one,a,1,2
one,b,3,4
one,c,5,6
one,d,7,8
two,a,9,10
two,b,11,12
two,c,13,14
two,d,15,16
"""
# 如果想读取成分层索引，则需要为index_col参数传入一个包含列序号或列名的列表
df = pd.read_csv('d:/csv_mindex.csv', index_col=['key1','key2'])


"""
对于一些更复杂，或者说更不规整的文件，比如分隔符
不固定的情况，需要有更多的处理技巧
，比如下面这个文件，以不同数量的空格
分隔
"""
result = pd.read_table(r"D:\pycharmcode\第十一次学习\2.txt", sep='\s+')

# 使用skiprows来跳过数据中的指定行
result1 = pd.read_table(r"D:\pycharmcode\第十一次学习\2.txt", sep='\s+',skiprows =[0,2,3])

## read_csv会将缺失值自动读成NaN
pd.isnull(result)

## 可以额外指定na_values参数，将某些值也当作缺失值来看待
result = pd.read_table(r"D:\pycharmcode\第十一次学习\2.txt",na_values=[1.100491])
print(result)

## 甚至可以对不同的列，指定不同的缺失值标识
f = {'message':['foo','NA'],'something':['two']}
result = pd.read_csv('d:/ex5.csv',na_values=f)

"""
下表列出了read_csv和read_table函数的一些常用参数：

参数	说明
path	文件路径
sep	指定分隔符或正则表达式
header	用作列名的行号
index_col	用作行索引的列名或列号
names	结果的列名列表
skiprows	从起始处，需要跳过的行
na_values	需要用NaN替换的值
comment	在行结尾处分隔注释的字符
parse_dates	尝试将数据解析为datetime，默认是False
keep_date_col	如果连接列到解析日期上，保留被连接的列，默认False
converters	包含列名称映射到函数的字典
dayfirst	解析非明确日期时，按国际格式处理
date_parser	用于解析日期的函数
nrows	从文件开头处读入的行数
iterator	返回一个TextParser对象，用于零散地读入文件
chunksize	用于迭代的块大小
skip_footer	忽略文件尾部的行数
verbose	打印各种解析器输出的信息
encoding	文本编码，比如UTF-8
thousands	定义千位分隔符，例如逗号或者圆点
"""






"""
分块读取
  当我们处理大型文件的时候，读入文
  件的一个小片段或者按小块遍历文件
  是比较好的做法

"""
# 即使是大文件，最多也只会显式10行具体内容
pd.options.display.max_rows = 10

# 或者使用nrows参数，指明从文件开头往下只读n行
result = pd.read_csv('d:/ex6.csv',nrows=5)

# 或者指定chunksize作为每一块的行数，分块读入文件
chunker = pd.read_csv('d:/ex6.csv', chunksize=1000)




"""
写出数据
  既然有读，必然有写
  可以使用DataFrame的to_csv方法，
  将数据导出为逗号分隔的文件
   result.to_csv(r"xx/xx/xx")

"""
#  Series的写入方式也是一样的
dates = pd.date_range('21/2/2022', periods=7)
ts = pd.Series(np.arange(7), index=dates)

# 写入文件中
ts.to_csv('d:/tseries.csv')






"""
EXCEL文件

"""
# 打开excel文件
xlsx = pd.ExcelFile(r"D:\pycharmcode\第十一次学习\data2.xlsx")

# 读取指定的表
pd.read_excel(xlsx, 'Sheet1')

pd.read_excel(xlsx, 'Sheet2')


## 写回到excel文件
# 生成文件
writer = pd.ExcelWriter('d:/ex2.xlsx')
# 写入
df.to_excel(writer, 'Sheet1')
# 关闭文件
writer.save()

# 快捷操作：
df.to_excel('d:/ex3.xlsx')















"""
Web交互
  很多网站都有公开的API，通过JSON或
  者其它格式提供数据服务。我们可以
  利用Python的requests库来访问这
  些API。
  ---获取Github上最新的30条关于
  pandas的问题为例
"""
# 导入包
import requests
# 要访问的url
url = 'https://api.github.com/repos/pandas-dev/pandas/issues'

# 访问页面，需要等待一会
response = requests.get(url)
# 解析为json格式
data = response.json()
# 查看第一个问题的标题
data[0]["title"]
data[0]


## 可以将data直接传给DataFrame，并提取感兴趣的字段
issues = pd.DataFrame(data, columns= ['number','title','labels','state'])




"""
数据库交互
   在实际使用场景中，大部分数据并不是存储在文本或者Excel文件中的，而是一些基于SQL语言的关系型数据库中，比如MySQL。

   从SQL中将数据读取为DataFrame对象是非常简单直接的，pandas提供了多个函数用于简化这个过程。

   下面以Python内置的sqlite3标准库为例，介绍一下操作过程。
"""
import sqlite3
# 编写一条创建test表的sql语句
query = """
     ...: CREATE TABLE test
     ...: (a VARCHAR(20), b VARCHAR(20), c REAL, d integer);"""
# 创建数据库文件，并连接
con = sqlite3.connect("mydata.sqlite")
# 执行sql语句
con.execute(query)
# 提交事务
con.commit




# 两个人和一只老鼠的信息
data = [('tom', 'male',1.75, 20),
  ('mary','female',1.60, 18),
  ('jerry','rat', 0.2, 60)]

# 再来一条空数据
stmt = "INSERT INTO test VALUES(?,?,?,?)"

# 执行多个语句
con.executemany(stmt,data)

# 再次提交事务
con.commit()


# 执行查询语句
cursor = con.execute('select * from test')

# 获取查询结果
rows = cursor.fetchall()

cursor.description

pd.DataFrame(rows,columns= [x[0] for x in cursor.description])


###  来个简单的
# 从通用的SQLAlchemy连接中轻松地读取数据
import sqlalchemy as sqla

# 创建连接
db = sqla.create_engine('sqlite:///mydata.sqlite')
# 查询数据并转换为pandas对象
pd.read_sql('select * from test', db)



"""
删除缺失值
   Pandas中，使用numpy.nan标识缺失
   值，在打印的时候，经常以空字符串
   、NA、NaN、NULL等形式出现。
   Python内置的None值也被当作缺失
   值处理。但合法的缺失值只有NaN
   和None

   None被看作一个object对象，需要消耗更多的资源，处理速度更慢。不支持一些数学操作，因为None+数字是错误的语法。很多时候None会自动转换成NaN。
NaN是float64类型，虽然名字叫做‘不是一个数’，但却属于数字类，可以进行数学运算不会报错，虽然所有和它进行计算的最终结果依然是NaN。它的运算速度更快，还支持全局性的操作。

    缺失值处理方法:
        dropna：删除缺失值
        fillna: 用某些值填充缺失的数据或使用插值方法（比如ffill\bfill）
        isnull：判断哪些值是缺失值，返回布尔
        notnull：isnull的反函数
"""
# 导入惯例
from numpy import nan as NA
s = pd.Series([1, NA, 3.5, NA, 7])
print(s)

# 本质上就是把缺失值删除
s.dropna()
# 等同于上面的操作
s[s.notnull()]


## dropna默认情况下会删除包含缺失值的行
df = pd.DataFrame([[1, 6.5, 3],[1, NA, NA],[NA, NA, NA],[NA, 6.5,3]])
# 只剩1行了
df.dropna()

# 只将整行都是缺失值的删除
df.dropna(how='all')
# 新增一列全是缺失值
df[4] = NA
# 指定以列的形式删除
df.dropna(axis=1, how='all')
df[df.columns[0:2]]

# 取hang行
df.loc[[0,2]]
df.iloc[0:2]
df.iloc[[0,2]]



"""
删除重复值

"""
import pandas as pd
df = pd.DataFrame({'k1':['one','two']*3 + ['two'], 'k2':[1,1,2,3,3,4,4]})

# 使用duplicated方法判断各行是否有重复
df.duplicated()
# 使用drop_duplicates方法将重复行删除
df.drop_duplicates()

# 指定根据某列的数据进行去重判断和操作
df['v1'] = range(7)
df.drop_duplicates(['k1'])
df.drop_duplicates(['k2'])

# 默认情况下都是保留第一个观察到
# 的值，如果想保留最后一个，可以使用参数keep='last'
df.drop_duplicates(["k1","k2"],keep = "last")




"""
替换
  replace
"""
from numpy import nan as NA
import numpy as np
df = pd.DataFrame(np.random.randint(12,size=(4,3)))

# 将4替换为缺失值
df.replace(3,NA)

# 将3和4都替换为缺失值
df.replace([3,8],NA)

# 3和4分别替换为缺失值和0
df.replace([3,8], [NA,0])

# 参数的字典形式
df.replace({3:NA,8:0})



"""
重命名轴索引
  与Series对象类似，轴索引也有一个
   map的方法
"""
df = pd.DataFrame(np.arange(12).reshape((3, 4)),
                  index=['Ohio', 'Colorado', 'New York'],
                   columns=['one', 'two', 'three', 'four'])

# 截取前3个字符并大写
transform = lambda x: x[:4].upper()
print(transform)

# map的结果
df.index.map(transform)

#用结果修改原来的index
df.index = df.index.map(transform)
print(df)

## 可以使用rename方法修改索引

df.rename(index=str.title, columns=str.upper)
# 原值未变
df


# 使用字典的方式
df.rename(index={'OHIO': 'INDIANA'},
         columns={'three': 'peekaboo'})

# 使用inplace=True可以原地修改数据集






"""
离散化和分箱
     cut方法---离散化，就是将连续值转换为一个个区间内，形成一个个分隔的‘箱子’。
     分箱的区间通常是左开右闭的，如果想变成左闭右开，请设置参数right=False
"""
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]

bins = [18,25,35,60,100]
cats = pd.cut(ages,bins)

print(cats)

# cats是一个特殊的Categorical对象
# cats包含一系列的属性
cats.codes
cats.categories
cats.describe

# 各个箱子的数量
pd.value_counts(cats)

d =np.random.rand(20)
# 切割成四份  精度限制在两位
pd.cut(d, 4, precision=2)

###  qcut，它是使用样本的分位数来分割的
data = np.random.randn(1000)
cats = pd.qcut(data,4)
# 各箱子中的元素个数相同
pd.value_counts(cats)

pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])


















"""
检测和过滤
   检测和过滤异常值是数据清洗过程中非常重要的一步

"""
df = pd.DataFrame(np.random.randn(1000, 4))
df.describe()

# 找出第二列数据中绝对值大于3的元素
col = df[2]
col[np.abs(col)>3]

# 要选出所有行内有值大于3或小于-3的行
df[(np.abs(df)>3).any(1)]

# 还可以将绝对值大于3的数分别设置为+3和-3
# np.sign(x)函数，这个函数根据x的符号分别生成+1和-1
df[np.abs(df) > 3] = np.sign(df) * 3
(np.sign(df)*3).head()










"""
随机和抽样
      有时候，我们需要打乱原有的数据顺序，让数
      据看起来像现实中比较混沌、自然的样子。
      这里推荐一个permutation操作，它来自
      numpy.random，可以随机生成一个序列



"""

# 5个数
order = np.random.permutation(5)

df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
# 取出df中通过permutation之后的数据
df.take(order)
df.iloc[order]

### 从原样本集合中抽取一部分形成新的样本集合，分重复抽样和不重复抽样
   # n=3：指定从原来数据中抽取3行
   # n=10：弹出异常，因为原数据不够10行
   # replace=True：可以重复抽取，这样10行可以，因为有重复
df.sample(n=3)

df.sample(n=10)

df.sample(n=10,replace=True)






"""
字符串操作
     Python内置的字符串操作和re正则模块可以
     帮我们解决很多场景下的字符串操作需求。
     但是在数据分析过程中，它们有时候比较
     尴尬
     Pandas为这一类整体性的操作，提供了
     专门的字符串函数，帮助我们跳过缺失值
     等异常情况，对能够进行操作的每个元素
     进行处理
"""
dic= {'one':'feixue', 'two':np.nan, 'three':'tom', 'five':'jerry@film'}
s = pd.Series(dic)

s.str.upper()

"""
  Series的str属性，在它的基础上甚至可以使用正则表达式的函数
    cat :粘合字符串
    contains：是否包含的判断
    count：计数
    extract：返回匹配的字符串组
    endswith：以xx结尾判断
    startswith：以xx开始判断
    findall：查找
    get：获取
    isalnum：类型判断
    isalpha：类型判断
    isdecimal：类型判断
    isdigit：类型判断
    islower：是否小写
    isnumeric：类型判断
    isupper：是否大写
    join：连接
    len：长度
    lower：小写
    upper：大写
    match：匹配
    pad：将空白加到字符串的左边、右边或者两边
    center：居中
    repeat：重复
    replace:替换
    slice：切片
    split：分割
    strip：脱除
    lstrip：左脱除
    rstrip：右脱除

"""








"""分层索引
   Pandas提供了Panel和Panel4D对象解决三维和四维数据的处理需求
   但更常用的还是分层索引
   也就是如何用Series、DataFrame处理
   三维、四维等等高维度的数据

"""
s = pd.Series(np.random.randn(9), index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'], [1, 2, 3, 1, 3, 1, 2, 2, 3]])
s.index

# MultiIndex就是一个分层索引对象，在打印的时候会进行规整的美化
s["b"]
s['b':'c']
s.loc[['b','d']]
# 这样不可以 s.loc['b','d']
# 但是这样可以
s.loc["b",1]

# 或者这样
s.loc[:, 2]



tup = [('beijing',2000),('beijing',2019),
      ('shanghai',2000),('shanghai',2019),
       ('guangzhou',2000),('guangzhou',2019)]
values = [10000,100000,6000,60000,4000,40000]

# 利用元组生成MultiIndex
index = pd.MultiIndex.from_tuples(tup)
# 提供一个MultiIndex作为索引
sss = pd.Series(values, index=index)
"""
更多的创建MultiIndex的方法还有：

从列表：pd.MultiIndex.from_arrays([['a','a','b','b'],[1,2,1,2]])
从元组：pd.MultiIndex.from_tuples([('a',1),('a',2),('b',1),('b',2)])
笛卡儿积：pd.MultiIndex.from_product([['a','b'],[1,2]])
直接构造：pd.MultiIndex(levels=[['a','b'],[1,2]],labels=[[0,0,1,1],[0,1,0,1]])
"""
# 可以使用unstack方法将数据在DataFrame中重新排列
s.unstack()
# 反操作stack
s.unstack().stack()

df = pd.DataFrame(np.arange(12).reshape((4, 3)),
                      index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                    columns=[['Ohio', 'Ohio', 'Colorado'],
                             ['Green', 'Red', 'Green']])
df.columns.names = ['state','color']
df['Ohio']






"""
分层索引进阶
    sort_index(level=1)意思是在第2个层级上进行索引的排序。
    swaplevel(0, 1)的意思是将第0层和第1层的行索引进行交换。

"""
# 1.1 重排序和层级排序
  # Pandas的swaplevel方法用于这一功能，层级发生变更，但原数据不变
df.swaplevel('key1', 'key2')

# 1.2 层级的汇总统计
df.sum(level='key2')

# 1.3 使用DataFrame的列进行索引
df= pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                       'c': ['one', 'one', 'one', 'two', 'two',
                               'two', 'two'],
                       'd': [0, 1, 2, 0, 1, 2, 3]})
# 不连后面
df2 = df.set_index(['c','d'])
# 连后面
df.set_index(['c','d'],drop=False)
# set_index(['c','d'])，将c列和d列变成了分层的行索引
# drop=False则保留了原来的列数据
# reset_index是set_index的反向操作

# 1.4 分层索引的取值与切片
tup = [('beijing',2000),('beijing',2019),
         ('shanghai',2000),('shanghai',2019),
         ('guangzhou',2000),('guangzhou',2019)]
values = [10000,100000,6000,60000,4000,40000]
index = pd.MultiIndex.from_tuples(tup)
s = pd.Series(values, index=index)

s['beijing',2019]
s[s>5000]








"""
合并连接
    pandas.merge: 根据一个或多个键进行连接。类似SQL的连接操作
            merge方法将两个pandas对象连接在一起，类似SQL的连接操作。默认情况下，它执行的是内连接，也就是两个对象的交集。通过参数how，还可以指定外连接、左连接和右连接。参数on指定在哪个键上连接，参数left_on和right_on分别指定左右对象的连接键。

                外连接：并集
                内连接：交集
                左连接：左边对象全部保留
                右连接：右边对象全部保留

    pandas.concat:使对象在轴向上进行粘合或者‘堆叠’
    ombine_first:将重叠的数据拼接在一起，使用一个对象中的值填充另一个对象中的缺失值
"""
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                        'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                        'data2': range(3)})
# 默认内链接，并智能地查找连接的键
pd.merge(df1,df2)

# 最好是显式地指定连接的键
pd.merge(df1,df2,on='key')
# 外连接
pd.merge(df1, df2, how='outer')

# 左连接
pd.merge(df1, df2, how='left')
#右连接
pd.merge(df1, df2, how='right')











"""
粘合与堆叠
  一、轴向连接
  concat方法可以实现对象在轴向的的粘合或者堆叠。
   对于DataFrame，默认情况下都是按行往下合并的，当然也可以设置axis参数
  二、联合叠加
   有这么种场景，某个对象里缺失的值，拿另外一个对象的相应位置的值来填补。在Numpy层面，可以这么做


"""
import pandas as pd
# 一、轴向连接
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
# 要以列表的方式提供参数
pd.concat([s1,s2,s3])

# 按人家的要求做
pd.concat([s1,s2,s3],axis=1,sort=True)
# 对于DataFrame，默认情况下都是按行往下合并的，当然也可以设置axis参数
df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                       columns=['one', 'two'])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                        columns=['three', 'four'])
pd.concat([df1, df2], sort=True)
pd.concat([df1, df2], axis=1, sort=True)

# 二、联合叠加
a = pd.Series([np.nan, 2.5, 0, 3.5, 4.5, np.nan],
                  index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series([0, np.nan, 2.1, np.nan, np.nan, 5], index=list('abcdef'))

# np.where(pd.isnull(a), b, a)，
# 这一句里，首先去pd.isnull(a)种
# 判断元素，如果是True，从b里拿数据，否则从a里拿，得到最终结果
np.where(pd.isnull(a), b, a)

# Pandas为这种场景提供了一个专门的combine_first
b.combine_first(a)








"""
重塑
   对表格型数据进行重新排列的操作，被称作重塑或者透视。
   使用多层索引进行重塑主要有stack和unstack操作，前面有介绍过
"""
import numpy as np
df = pd.DataFrame(np.arange(6).reshape(2,3),
   index=pd.Index(['河南','山西'], name='省份'),
   columns=pd.Index(['one','two','three'],name='number'))

result = df.stack()
result = df.unstack()

# 可以传入一个层级序号或名称来拆分不同的层级
result.unstack(0)
result.unstack('省份')

# 如果层级中的所有值并未有包含于每个子分组中，拆分可能会导致缺失值的产生
s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])

# 保留缺失值
data2.unstack().stack(dropna=False)



# 而在DataFrame对象拆堆时，被拆的层级会变成结果中最低的层级
df = pd.DataFrame({'left': result, 'right': result + 5},
                      columns=pd.Index(['left', 'right'], name='side'))

df.unstack('省份')
















"""
matplotlib
  通过plt.style.available查看可用的样式
  设置样式方法：plt.style.use('classic')
  常用样式：seaborn
  plt.show()
  plt.draw()强制刷新
"""

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2*np.pi*t)

plt.plot(t,s)

c = np.cos(2*np.pi*t)
plt.rcParams['lines.linewidth'] = '4'
plt.plot(t,c)

# 通过plt.style.available查看可用的样式
plt.style.available


"""
保存图形
    通过savefig()方法
    下面是savefig方法的参数说明：

        fname：文件路径或文件对象，根据扩展名推断文件格式
        dpi：分辨率，默认100
        format： 指定文件格式
        bbox_inches： 要保存的图片范围。‘tight’表示去掉周边空白。
        facecolor：子图之外的背景颜色，默认白色
        edgecolor：边界颜色
"""
x = np.linspace(0,10,100)
fig = plt.figure()
plt.plot(x,np.sin(x),'-')
plt.plot(x,np.cos(x),'--')
fig.savefig('d:/my_fig.png')

# 通过IPython的Image来显示文件内的图像
from IPython.display import Image
Image(r"D:\pycharmcode\第十一次学习\2.png")

# 产看系统支持的图片格式
fig.canvas.get_supported_filetypes()

"""
savefig方法有一些可定制的参数，比如你想得到一个600dpi的图片，并且尽量少的空白：
"""
plt.savefig('image_name.png', dpi=600,bbox_inches='tight')

# savefig也可以写入到文件对象中，比如BytesIO
from io import BytesIO
buffer = BytesIO()
plt.savefig(buffer)
plot_data = buffer.getvalue()



"""
两种图画接口
     1.MATLAB接口
        这种接口最重要的特性是有
        状态的，他会持续跟踪当前
        的图形和坐标轴，所有plt
        命令都可以使用。你可以
        使用plt.gcf()方法获取
        当前图形和plt.gca()获取
        当前坐标轴的具体信息。
        ---但是这种接口也有问题。
           比如，当创建第二个子图
           的时候，怎么才能回到第
           一个子图，并增加新内容呢？虽然也能实现，但方法比较复杂。而下面的方式则不存在这个问题。
     2.面向对象的接口
         这种方式可以适应更加复杂的
         场景，更好地控制你的图形。
         画图函数不再受到当前‘活动’
         图形或者坐标轴的限制，而变
         成了显式的Figure和Axes的
         方法。
"""
x = np.linspace(0,10,100) # 生成点列表
plt.figure() # 创建图形
plt.subplot(2,1,1)  # 行、列、子图编号
plt.plot(x,np.sin(x))
plt.subplot(2,1,2)
plt.plot(x,np.cos(x)) # 第二个子图



fig, ax = plt.subplots(2) # ax是包含两个Axes对象的数组
ax[0].plot(x,np.sin(x)) # 在每个对象上调用plot()方法
ax[1].plot(x,np.tan(x))




"""
使用中文
     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
其他方法：https://www.liujiangblog.com/course/data/256



"""








"""
画图：官方展示链接：https://matplotlib.org/gallery/index.html
线型图
   plot方法的核心是plot(x,y)，
   x表示横坐标值的序列，y表示x
   某个坐标对应的y值，实际上就
   是y=f(x)函数。当只提供y的时候，
   x默认使用0-n的整数序列。这里的
   序列必然是个有限的点集，而不是
   我们想象中的无穷个点组成一条线。
   如果你的点很稀疏，那么图形看起
   来就像折线，如果点很多，看起来
   就比较圆滑，形似曲线




"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(-10,10,200)
plt.plot(x, x**2)
plt.plot(x, x**3)

"""
matplotlib其实是一个相当底层的工具
，你可以从其基本组件中组装一个图标
、显示格式、图例、标题、注释等等。
Pandas在此基础上对绘图功能进行了
一定的封装，每个Series和DataFrame
都有一个plot方法，一定要区分pandas
的plot和matplotlib的plot方法。
"""
import pandas as pd
df = pd.DataFrame(np.random.randn(10,4).cumsum(0),columns=['A', 'B', 'C', 'D'],
                  index=np.arange(0, 100, 10))

df.plot()




"""
颜色线型和标记
    使用color参数可以指定线条的颜色
     下面是常用的颜色：

        蓝色： 'b' (blue)
        绿色： 'g' (green)
        红色： 'r' (red)
        蓝绿色(墨绿色)： 'c' (cyan)
        红紫色(洋红)： 'm' (magenta)
        黄色： 'y' (yellow)
        黑色： 'k' (black)
        白色： 'w' (white)
"""
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10,10,200)


plt.plot(x, np.sin(x - 0), color='blue')        # 英文字符串
plt.plot(x, np.sin(x - 1), color='c')           # 颜色代码(rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75')        # 0~1之间的灰度
plt.plot(x, np.sin(x - 3), color='#FFDD44')     # 十六进制形式
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB元组
plt.plot(x, np.sin(x - 5), color='chartreuse'); # HTML颜色


"""
可以使用linestyle参数指定线型。线型有两种表示方式：一是英文单词，二是形象符号。

常用的线型和符号对应：

        实线：solid(- )
        虚线：dashed(--)
        点划线：dashdot(-.)
        实点线：dotted(:)
可以通过marker参数来设置标记的类型：
        更多标记类型：

            '.' 实点标记
            ',' 像素标记
            'o' 圆形标记
            'v' 向下三角符号
            '^' 向上三角符号
            '<' 向左三角符号
            '>' 向右三角符号
            '1' 三叉星符号
            '2' 三叉星符号
            '3' 三叉星符号
            '4' 三叉星符号
            's' 方形
            'p' 五边形
            '*' 星型
            'h' 六边形1
            'H' 六边形2
            '+' 加号
            'x' 叉叉
            'D' 钻石形状
            'd' 菱形
            '|' 竖条
            '_' 横条
"""
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted')

x = np.linspace(0,10,10)
plt.plot(x, x + 0, marker='.')
plt.plot(x, x + 1, marker=',')
plt.plot(x, x + 2, marker='o')
plt.plot(x, x + 3, marker='+')

# 此外，还有一种更便捷的做法，那就是
# 组合颜色、线型和标记的设置。
# 三者顺序有时可以随意，
# 但最好使用‘颜色+标记+线型’的顺序

plt.plot(x, x + 0, 'go-')  # 绿色实线圆点标记
plt.plot(x, x + 1, 'c--') # 青色虚线
plt.plot(x, x + 2, '-.k*') # 黑色点划线星型标记
plt.plot(x, x + 3, ':r');  # 红色实点线
"""
对于plot()方法，大部分可配置的参数如下：

参数	取值范围	说明
alpha	0-1	透明度
color或c	颜色格式	设置线条颜色
label	字符串	为图形设置标签
linestyle或ls	可用线型	设置线条风格
linewidth或lw	数值	线宽
marker	可用标记	标记
markeredgecolor或mec	颜色	标记的边缘颜色
markeredgewidth或mew	数值	标记的边缘宽度
markerfacecolor或mfc	颜色	标记的颜色
markersize或ms	数值	标记的大小
solid_capstyle	butt、round、projecting	实线的线端风格
solid_joinstyle	miter、round、bevel	实线的连接风格
drawstyle	default、steps、steps-pre、steps-mid、steps-post	连线的规则
visible	True、False	显示或隐藏
xdata	np.array	主参数x的输入
ydata	np.array	主参数y的输入

"""













"""
坐标轴上下限
      使用plt.xlim()
      和plt.ylim()来调整上下限的值：
"""
x = np.linspace(0,10,100)
plt.plot(x,np.sin(x))
plt.xlim(-1,11)
plt.ylim(-1.5,1)

# 坐标轴逆序显示
plt.plot(x,np.sin(x))
plt.xlim(11,-1)
plt.ylim(1.5,-1.5)

# 使用plt.axis()方法设置坐标轴的
# 上下限（注意区别axes和axis），
# 参数方式是[xmin, xmax, ymin, ymax]
plt.plot(x,np.sin(x))
plt.axis([-1,11,-1.5,1.5])


# axis的作用不仅于此，
# 还可以按照图形的内容自动收缩坐标轴，
# 不留空白

plt.plot(x,np.sin(x))
plt.axis('tight')

plt.plot(x,np.sin(x))
plt.axis('off')
"""
plt.axis()更多类似的常用设置值有：

off：隐藏轴线和标签
tight：紧缩模式
equal：以1：1的格式显示，x轴和y轴的单位长度相等
scaled: 通过更改绘图框的尺寸来获得相同的结果
square: x轴和y轴的限制值一样
"""





"""
坐标轴刻度

"""
plt.plot(np.random.randn(1000).cumsum())

# 可以手动提供刻度值，并调整刻度的角度和大小
plt.plot(np.random.randn(1000).cumsum())
plt.xticks([0,250,500,750,1000],rotation=120, fontsize='large')
plt.yticks([-45,-35,-25,-15,0],rotation=30, fontsize='small')




"""
图题、轴标签和图例
    图题： plt.title()
    轴标签：plt.xlabel()、plt.ylabel()
    图例：plt.legend()

"""
plt.plot(x, np.sin(x),'-g',label='sin(x)')
plt.plot(x, np.cos(x),':b',label='cos(x)')
plt.plot(x,np.tan(x),'-.k*',label="tan(x)")
plt.title('a sin curve')
plt.xlabel("X")
plt.ylabel("sin(X)")
plt.legend()

"""
注意：大多数的plt方法都可以直接转换
成ax方法，比如plt.plot()
->ax.plot(),plt.legend()
->ax.legend()
    plt.xlabel() -> ax.set_xlabel()
    plt.ylabel() -> ax.set_ylabel()
    plt.xlim() -> ax.set_xlim()
    plt.ylim() -> ax.set_ylim()
    plt.title() -> ax.set_title()
在面向对象接口画图的时候，
不需要单独调用这些函数，
使用ax.set()方法一次性设置即可

"""
x = np.linspace(0,10,100)
ax = plt.axes()
ax.plot(x,np.sin(x))
ax.set(xlim=(0,10),ylim=(-2,2),xlabel='x',ylabel='sin(x)',title='a sin  plot')










"""
配置图题
   title标题方法，也有许多可以配置的参数：

    fontsize：字体大小，默认12，也可以使用xx-small....字符串系列
    fontweight：字体粗细，或者'light'、'normal'、'medium'、'semibold'、'bold'、 'heavy'、'black'。
    fontstyle： 字体类型，或者'normal'、'italic'、'oblique'。
    verticalalignment：垂直对齐方式 ，或者'center'、'top'、'bottom'、'baseline'
    horizontalalignment：水平对齐方式，可选参数：‘left’、‘right’、‘center’
    rotation：旋转角度
    alpha： 透明度，参数值0至1之间
    backgroundcolor： 背景颜色
    bbox：给标题增加外框 ，常用参数如下：
    boxstyle：方框外形
    facecolor：(简写fc)背景颜色
    edgecolor：(简写ec)边框线条颜色
    edgewidth:边框线条大小

"""
plt.title('A Title',fontsize='large',fontweight='bold') #设置字体大小与尺寸
plt.title('A Title',color='blue') #设置字体颜色
plt.title('A Title',loc ='left') #设置字体位置
plt.title('A Title',verticalalignment='bottom') #设置垂直对齐方式
plt.title('A Title',rotation=45) #设置字体旋转角度
plt.title('A Title',bbox=dict(facecolor='g', edgecolor='blue', alpha=0.65 )) #设置标题边框

# title标题方法的大部分参数也适用于xlabel和ylabel坐标轴标记方法，大家可以自行尝试。




"""
配置图例
     参数	说明
loc	图例的位置
prop	字体参数
fontsize	字体大小
markerscale	图例标记与原始标记的相对大小
markerfirst	如果为True，则图例标记位于图例标签的左侧
numpoints	为线条图图例条目创建的标记点数
scatterpoints	为散点图图例条目创建的标记点数
scatteryoffsets	为散点图图例条目创建的标记的垂直偏移量
frameon	是否显示图例边框
fancybox	边框四个角是否有弧度
shadow	控制是否在图例后面画一个阴影
framealpha	图例边框的透明度
edgecolor	边框颜色
facecolor	背景色
ncol	设置图例分为n列展示
borderpad	图例边框的内边距
labelspacing	图例条目之间的垂直间距
handlelength	图例句柄的长度
handleheight	图例句柄的高度
handletextpad	图例句柄和文本之间的间距
borderaxespad	轴与图例边框之间的距离
columnspacing	列间距
title	图例的标题
对于loc这个图例在坐标轴中的放置位置，有两种表示方法：数字或者字符串，其对应关系如下：

0: ‘best' ： 自动选择最适合的位置
1: ‘upper right'： 右上
2: ‘upper left'： 左上
3: ‘lower left'： 左下
4: ‘lower right'：右下
5: ‘right'：右
6: ‘center left'：左中
7: ‘center right'：右中
8: ‘lower center'：下中
9: ‘upper center'： 上中
10: ‘center'：中间
设置字体大小的参数fontsize可以使用整数或者浮点数，以及字符串‘xx-small’、 ‘x-small’、 ‘small’、‘medium’、 ‘large’、 ‘x-large’和‘xx-large’
"""
plt.legend(loc='best',frameon=False) #去掉图例边框
plt.legend(loc='best',edgecolor='blue') #设置图例边框颜色
plt.legend(loc='best',facecolor='blue') #设置图例背景颜色,若无边框,参数无效
plt.legend(loc='best',title='figure') #去掉图例边框
plt.legend(loc='upper left', ncol=2, frameon=False) # 分两列显示，在左上角
plt.legend(fancybox=True,framealpha=1, shadow=True, borderpad=1)





lines = []
styles= ['-', '--','-.',':']
x = np.linspace(0,10,1000)

for i in range(4): # 制造四条sin曲线
    lines += plt.plot(x, np.sin(x-i*np.pi/2), styles[i])
plt.axis('equal')
# 生成第一个图例，并保存引用
leg = plt.legend(lines[:2], ['line A', 'line B'], loc=1,frameon=False)
# 生成第二个图例，这会让第一个图例被抹去
plt.legend(lines[2:], ['line C', 'line D'], loc=4,frameon=False)
# gca方法获取当前坐标轴，再使用它的`add_artist`方法将第一个图例重新画上去
plt.gca().add_artist(leg)












"""
颜色条
    在Matplotlib中，颜色条是一个独立的坐标轴，可以指明图形中的颜色的含义
     实际上在plt.cm中有大量可选的颜色配置方案。具体使用哪种方案，要根据你的需求和你对美术的修养。一般情况下，你只需要关注三种类型的配色方案：

        顺序配色：由一组连续的颜色构成，比如binary和viridis
        互逆色：通常由互补的颜色构成，比如RdBu或者PuOr
        定性配色：随机顺序的一组颜色，比如rainbow或jet
        这些配色方案的名字很多都是缩写组合，一定要注意字母大小写。


"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,1000)
I = np.sin(x)* np.cos(x[:,np.newaxis])
I.shape

plt.imshow(I)
plt.colorbar()


plt.imshow(I, cmap='jet')
plt.colorbar()
plt.imshow(I, cmap='RdBu')
plt.colorbar()


speckles = np.random.random(I.shape) < 0.01  # 制造bool噪点
I[speckles] = np.random.normal(0,3,np.count_nonzero(speckles)) # 生成噪点

plt.figure(figsize=(10,3.5))
plt.subplot(1,2,1)  # 多子图模式
plt.imshow(I, cmap='RdBu')
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both') # 颜色条变成两端尖
plt.clim(-1,1)  # 限制颜色显示范围

"""
颜色条默认都是连续的，但有时候你
可能也需要使用离散的颜色数据，
最简单的方法就是使用
plt.cm.get_cmap()方法，将适当的
颜色方案和需要的区间数量作为参数
传递进去即可
"""
plt.imshow(I, cmap=plt.cm.get_cmap('Blues',6))
plt.colorbar()
plt.clim(-1,1)








"""
文本、箭头和注释
    很多时候，光是图像不足以表达所有的内容，需要一些说明性的文字来辅助。

   在Matplotlib中，使用plt.text方法为图例添加文字
   plt.text方法的签名如下：

plt.text(x, y, s, fontdict=None, withdash=False, **kwargs)
        下面是常用的参数说明：

        x,y:坐标值，文字放置的位置
        string:文字内容字符串
        size:字体大小
        alpha:设置字体的透明度
        family: 设置字体
        style:设置字体的风格
        wight:字体的粗细
        verticalalignment：垂直对齐方式，缩写为va。可用值 ‘center’ | ‘top’ | ‘bottom’ | ‘baseline’。
        horizontalalignment：水平对齐方式 ，缩写为ha。可用值‘center’ | ‘right’ | ‘left’
        xycoords:选择指定的坐标轴系统
        bbox给标题增加外框 ，常用参数如下：
        boxstyle:方框外形
        facecolor:(简写fc)背景颜色
        edgecolor:(简写ec)边框线条颜色
        edgewidth:边框线条大小
        其它未列出
"""
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,10,100)
plt.plot(x, np.sin(x))
plt.text(5,0.5,'this is a sin(x) curve',ha='center',va='center')




plt.text(0.3, 0.3, "hello", size=50, rotation=30.,ha="center", va="center",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
plt.text(0.8, 0.8, "world", size=50, rotation=-40.,ha="right", va="top",bbox=dict(boxstyle="square",ec=(1., 0.2, 0.5),fc=(1., 0.3, 0.8),))


## 很多时候，文本是以数学公式出现的
x = np.linspace(0,10,100)
plt.plot(x, np.sin(x))
plt.title(r'$\alpha_i > \beta_i$', fontsize=20)
plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
plt.text(3, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',fontsize=20)


# 一般使用plt.annotate()方法来实现箭头和注释的功能
fig, ax = plt.subplots()
x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

"""
annotate(s='str' ,xy=(x,y) ,xytext=(l1,l2) ,arrowprops=dict(...)..)
    主要参数说明：

    s：注释文本内容
    xy：被注释对象的坐标位置，实际上就是图中箭头的箭锋位置
    xytext： 具体注释文字的坐标位置
    xycoords：被注释对象使用的参考坐标系
    extcoords：注释文字的偏移量
    arrowprops：可选，增加注释箭头
       arrowprops参数的基本配置项
          width：箭头宽度，以点为单位
            frac：箭头头部所占据的比例
            headwidth：箭头底部的宽度，以点为单位
            shrink：移动提示，并使其离注释点和文本一些距离
            **kwargs：matplotlib.patches.Polygon的任何键，例如facecolor
 https://matplotlib.org/users/annotations.html#plotting-guide-annotation
https://matplotlib.org/examples/pylab_examples/annotation_demo2.html
具体例子的展示
"""
fig = plt.figure(1, figsize=(8, 5))
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 5), ylim=(-4, 3))

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = ax.plot(t, s, lw=3)


ax.annotate('straight',
            xy=(0, 1), xycoords='data',
            xytext=(-50, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))


ax.annotate('arc3,\nrad 0.2',
            xy=(0.5, -1), xycoords='data',
            xytext=(-80, -60), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=.2"))

ax.annotate('arc,\nangle 50',
            xy=(1., 1), xycoords='data',
            xytext=(-90, 50), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc,angleA=0,armA=50,rad=10"))

ax.annotate('arc,\narms',
            xy=(1.5, -1), xycoords='data',
            xytext=(-80, -60), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc,angleA=0,armA=40,angleB=-90,armB=30,rad=7"))

ax.annotate('angle,\nangle 90',
            xy=(2., 1), xycoords='data',
            xytext=(-70, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=90,rad=10"))

ax.annotate('angle3,\nangle -90',
            xy=(2.5, -1), xycoords='data',
            xytext=(-80, -60), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

ax.annotate('angle,\nround',
            xy=(3., 1), xycoords='data',
            xytext=(-60, 30), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=90,rad=10"))

ax.annotate('angle,\nround4',
            xy=(3.5, -1), xycoords='data',
            xytext=(-70, -80), textcoords='offset points',
            size=20,
            bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=-90,rad=10"))

ax.annotate('angle,\nshrink',
            xy=(4., 1), xycoords='data',
            xytext=(-60, 30), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(arrowstyle="->",
                            shrinkA=0, shrinkB=10,
                            connectionstyle="angle,angleA=0,angleB=90,rad=10"))


















"""
散点图
   用plt.scatter画散点图
   plt.scatter(x, y, marker='o')
   主要参数说明:

        x，y：输入数据
        s：标记大小，以像素为单位
        c：颜色
        marker：标记
        alpha：透明度
        linewidths：线宽
        edgecolors ：边界颜色

scatter的更多内容请参考官网：https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
"""
# 用plt.plot画散点图
x = np.linspace(0,10,30)
y = np.sin(x)
plt.plot(x,y,'bo', ms=5)




rng = np.random.RandomState(42)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000* rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3)
plt.colorbar() # 绘制颜色对照条


from sklearn.datasets import load_iris
iris = load_iris()
iris
iris.data  # 查看一下
iris.target  # 查看一下
iris.feature_names  # 查看一下
features = iris.data.T   # 转置
plt.scatter(features[0],features[1],alpha=0.2, s=100*features[3],c=iris.target)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);














"""
直方图
  使用hist方法来绘制直方图
   绘制直方图，最主要的是一个
   数据集data和需要划分的
   区间数量bins，
   另外你也可以设置一些颜色、类型参数

  histtype直方图的类型，
  可以是
  'bar'、 'barstacked'、'step'和'stepfilled'。
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# 创建数据
mu = 100  # 分布的均值
sigma = 15  # 分布标准差
x = mu + sigma * np.random.randn(400) # 生成400个数据

num_bins = 50  # 分50组
plt.hist(x, num_bins, density=0, )

plt.xlabel("smarts")
plt.ylabel('Probability density')
plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')


plt.hist(np.random.randn(1000), bins=30,density=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')

"""
直方对比图

"""
x1 = np.random.normal(0,0.8,1000)
x2 = np.random.normal(-2,1,1000)
x3 = np.random.normal(3,2,1000)

params = dict(histtype='stepfilled', alpha=0.3, density=True,bins=40)
plt.hist(x1, **params)  # 以字典的形式提供参数
plt.hist(x2, **params)  # 在同一个子图中绘制，颜色会自动变化
plt.hist(x3, **params)

"""
还可以使用hist2d方法绘制二维的直方图
"""
mean = [0,0]  # 忽略数据的创建过程
cov = [[1,1],[1,2]]
x,y = np.random.multivariate_normal(mean, cov,10000).T

plt.hist2d(x,y,bins=30,cmap='Blues')  #以蓝色为基调
cb = plt.colorbar()  # 插入颜色条
cb.set_label('counts in bin')  # 设置颜色条的标签

"""
hist2d是使用坐标轴正交的方块分割区
域，还有一种常用的方式是正六边形
也就是蜂窝形状的分割。
Matplotlib提供的plt.hexbin就是
满足这个需求的
"""
plt.hexbin(x,y,gridsize=30, cmap='Blues')
plt.colorbar(label='count in bin')












"""
条形图
    条形图，也称柱状图，看起来像直
    方图，但完是两码事。条形图根据
    不同的x值，为每个x指定一个
    高度y，画一个一定宽度的条形；
    而直方图是对数据集进行区间划分
    ，为每个区间画条形
"""
n = 12   # 12组数据
X = np.arange(n)
Y1 = (1 - X / n) * np.random.uniform(0.5, 1.0, n)  # 生成对应的y轴数据
Y2 = (1 - X / n) * np.random.uniform(0.5, 1.0, n)
plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')  # +号让所有y值变成正数
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white') # 负号让所有y值变成复数
# 加上数值
for x, y in zip(X, Y1):  # 显示文本
    plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    plt.text(x, -y - 0.05, '-%.2f' % y, ha='center', va='top')

plt.xlim(-0.5, n)
plt.ylim(-1.25, 1.25)

"""
将上面的代码稍微修改一下，
就可以得到下面的图形
即把对比图像放在同一个方向

"""
plt.bar(X, Y1, width=0.4, facecolor='lightskyblue', edgecolor='white')
plt.bar(X+0.4, Y2, width=0.4, facecolor='yellowgreen', edgecolor='white')

for x,y in zip(X,Y1):
    plt.text(x, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

for x,y in zip(X,Y2):
    plt.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

plt.xlim(-0.5,6)
plt.ylim(0,1.25)










"""
饼图
  通过pie方法，可以绘制饼图
  主要参数说明：

    x：输入的数据数组
    explode：数组，可选参数，默认为None。 用来指定每部分从圆中外移的偏移量。 例如：explode=[0,0.2,0,0]，第二个饼块被拖出。
    labels：每个饼块的标记
    colors：每个饼块的颜色
    autopct：自动标注百分比，并设置字符串格式
    shadow：是否添加阴影。
    labeldistance：被画饼标记的直径。
    startangle：从x轴逆时针旋转饼图的开始角度。
    radius：饼图的半径
    counterclock：指定指针方向，顺时针或者逆时针。
    center：图表中心位置。
"""
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
labels = '狗', '猫', '青蛙', '乌龟'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal') # 设置x轴和y轴均等









"""
误差线
   使用errorbar方法可以绘制误差线。
   errorbar方法的一些参数说明：

        yerr: 绘制垂直方向的误差线
        xerr:绘制水平方向的误差线
        fmt: 线条格式
        ecolor: 误差线的颜色
        elinewidth:误差线的宽度
        capsize: 误差线的长度
"""
x = np.linspace(0,10,50)
dy=0.8
y = np.sin(x) + dy*np.random.randn(50)
plt.plot(x,y)
plt.errorbar(x, y, yerr=dy, fmt='.k')

plt.errorbar(x, y, yerr=dy, fmt='ok',ecolor='lightgray',elinewidth=3, capsize=0)
















"""
等高线
  我们经常在二维图上用等高线来表示三维数据，Matplotlib提供了三个函数来实现这一功能：

        plt.contour: 绘制等高线
        plt.contourf: 绘制带有填充色的等高线
        plt.imshow: 显示图形
当图形中只使用一种颜色的时候，会使用虚线来表示负数，实线表示正数。
"""
def f(x, y):
    return np.sin(x)**10 + np.cos(10 +y*x)*np.cos(x)

x = np.linspace(0,5,50)
y = np.linspace(0,5,40)

X, Y = np.meshgrid(x, y) # 使用两个一维数组构建一个二维数组
Z = f(X, Y)

plt.contour(X, Y, Z, colors='blue')

"""
我们可以将数据范围等分，比如10份，然后设置camp参数定义一个线条颜色的配色方案，用不同的颜色表示等高线。
RdGy是红-灰配色方案(Red-Gray)。
Matplotlib有很多配色方案可选，
都在plt.cm模块里，
使用plt.cm.<tab>可以查看有哪些方案。
"""
plt.contour(X, Y, Z, cmap='RdGy')
plt.colorbar()

# 可以通过contourf方法进行填充(注意方法名多了个f)
plt.contourf(X, Y, Z, cmap='RdGy')
plt.colorbar()

"""
可以通过imshow方法，将二维数组渲染成渐变图
   imshow不支持使用x和y轴的数据设置网格，必须通过extend参数设置图形的坐标范围xmin、xmax，ymin、ymax
   imshow默认使用标准的图形数组定义，原点位于左上角，类似浏览器。而不是绝大多数等高线的左下角
   imshow会自动调整坐标轴的精度以适应数据显示。可以通过plt.axis(aspect='image')来设置x与y轴的单位。
"""
plt.imshow(Z, extent=[0,5,0,5], origin='lower',cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image')

"""
最后，我们还可以使用clabel方法，
将一幅背景半透明的彩色图与另一幅
坐标相同、带数据标签的等高线叠放
在一起，画出相当高端的图形：
"""
contours = plt.contour(X,Y,Z,3,colors='black') # 分3级
plt.clabel(contours,inline=True, fontsize=8)  # 内部带数值文字

plt.imshow(Z, extent=[0,5,0,5], origin='lower',cmap='RdGy', alpha=0.5)
plt.colorbar()
















"""
多子图
   很多时候，我们需要从多个角度对数据进行比较，在可视化上也是一样的。Matplotlib通过子图subplot的概念来实线这一功能
"""
"""
一、手动创建子图
     通过plt.axes函数可以创建基本子图
     默认情况下它会创建一个标准的坐标轴，并填满整张图
     这个参数是个列表形式，
     有四个值，从前往后，分
     别是子图左下角基点
     的x和y坐标以及子图的
     宽度和高度，数值的取值范围
     是0-1之间，画布左下角是
     （0，0），画布右上角是（1，1）

"""
ax1 = plt.axes()  # 使用默认配置，也就是布满整个画布
ax2 = plt.axes([0.65,0.65,0.2,0.2]) # 在右上角指定位置

"""
面向对象画图接口中有类似的fig.add_axes()方法
"""
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.5,0.8,0.4],xticklabels=[],ylim=(-1.2,1.2))
ax2 = fig.add_axes([0.1,0.1,0.8,0.4],ylim=(-1.2,1.2))

x = np.linspace(0,10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

"""
二、 plt.subplot方法
    subplot的方法接收三个整数参数，分别表示几行、几列、子图索引值。索引值从1开始，从左上角到右下角依次自增。
子图间距好像不太恰当，
可以使用plt.subplots_adjust方法进行调整，它接受水平间距hspace和垂直间距wspace两个参数

"""
for i in range(1, 7):  # 想想为什么是1-7
    plt.subplot(2,3,i)
    plt.text(0.5,0.5,str((2,3,i)), fontsize=16, ha='center')

# 同样的，面向对象接口也有fig.add_subplot()方法可以使用：
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1,7):
    ax = fig.add_subplot(2,3,i)
    ax.text(0.5,0.5,str((2,3,i)), fontsize=16, ha='center')

"""
三、 快速创建多子图
可以使用subplots()方法快速的创建多子图环境，并返回一个包含子图的Numpy数组。
通过sharex和sharey参数，
自动地去掉了网格内部子图的坐标刻度
等内容，实现共享，让图形看起来
更整齐整洁

"""
fig, ax = plt.subplots(2,3,sharex='col', sharey='row')
for i in range(2):
    for j in range(3):
        ax[i,j].text(0.5,0.5,str((2,3,i)), fontsize=16, ha='center')

# subplot()和subplots()两个方法在
# 方法名上差个字母s外，subplots的
# 索引是从0开始的

"""
四、GridSpec 复杂网格
    前面的子图其实都比较规整，如果想实现不规则的多行多列子图，可以使用plt.GridSpec方法
grid = plt.GridSpec(2,3,wspace=0.4,hspace=0.4) # 生成两行三列的网格

plt.subplot(grid[0,0]) # 将0，0的位置使用
plt.subplot(grid[0,1:]) # 同时占用第一行的第2列以后的位置
plt.subplot(grid[1,:2])
plt.subplot(grid[1,2])
"""
grid = plt.GridSpec(2,3,wspace=0.4,hspace=0.4) # 生成两行三列的网格

plt.subplot(grid[0,0]) # 将0，0的位置使用
plt.subplot(grid[0,1:]) # 同时占用第一行的第2列以后的位置
plt.subplot(grid[1,:2])
plt.subplot(grid[1,2])

"""
下面是一个使用plt.GridSpec方法
创建多轴频次直方图的例子：
"""
# 创建一些正态分布数据，这不是我们关心的内容
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

# 建立网格和子图
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])  # 注意切片的方式
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# 在主子图上绘制散点图
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)

# 在附属子图上绘制直方图
x_hist.hist(x, 40, histtype='stepfilled',
            orientation='vertical', color='gray')
x_hist.invert_yaxis() # 让y坐标轴的值由大到小，逆序

y_hist.hist(y, 40, histtype='stepfilled',
            orientation='horizontal', color='gray')

y_hist.invert_xaxis() # 可以尝试不要这行，看看结果












"""
patch
   想绘制一些常见的图形对象呢？
   比如圆形、矩形、三角等等。
   可以在matplotlib.patches
   中找到它们。
   使用mpl.patches.<tab>可以
   查看有哪些可用的patch。
   下面简要介绍如何使用
   ax的add_patch方法绘制patch
"""
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

# 注意下面三种形状的参数提供方式
rect = plt.Rectangle((0.2,0.75),0.4,0.15,alpha=0.3,color='b')
circ = plt.Circle((0.7,0.2), 0.15,)
pgon = plt.Polygon([[0.15,0.15],[0.35,0.4],[0.2,0.6]])

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)






"""
自定义坐标轴刻度
    Matplotlib图形对象具有层级关系。
    Figure对象其实就是一个盛放图形
    元素的盒子box，每个figure都会
    包含一个或多个axes对象，而每个
    axes对象又会包含其它表示图形
    内容的对象，比如xais和yaxis
    ，也就是x轴和y轴。
"""
# 主要刻度和次要刻度
ax = plt.axes(xscale='log', yscale='log')
"""
我们发现每个主要刻度都显示未一个较大的刻度线和标签，而次要刻度都显示为一个较小的刻度线，
并且不现实标签。

 每种刻度线都包含一个坐标轴定位器
 （locator）和格式生成器（
 formatter），可以通过下面的方法
 查看
"""
ax.xaxis.get_major_locator()
ax.xaxis.get_major_formatter()
ax.xaxis.get_minor_locator()
ax.xaxis.get_minor_formatter()

# 隐藏刻度与标签
# 落在locator和formatter这两大属性上

ax = plt.axes()
x = np.linspace(0,10,100)
ax.plot(np.sin(x))

ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())

#设置刻度数量
# 默认情况下，matplotlib会自动帮我们调节刻度的数量，但有时候也需要我们自定义刻度数量
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(4))
    axi.yaxis.set_major_locator(plt.MaxNLocator(4))



"""
风格样式展示
 追求一下整体的风格样式
 >>> plt.style.available
['bmh',
 'classic',
 'dark_background',
 'fast',
 'fivethirtyeight',
 'ggplot',
 'grayscale',
 'seaborn-bright',
 'seaborn-colorblind',
 'seaborn-dark-palette',
 'seaborn-dark',
 'seaborn-darkgrid',
 'seaborn-deep',
 'seaborn-muted',
 'seaborn-notebook',
 'seaborn-paper',
 'seaborn-pastel',
 'seaborn-poster',
 'seaborn-talk',
 'seaborn-ticks',
 'seaborn-white',
 'seaborn-whitegrid',
 'seaborn',
 'Solarize_Light2',
 'tableau-colorblind10',
 '_classic_test']
"""
def hist_and_lines():
    np.random.seed(10)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))

    x = np.linspace(0,10,100)
    ax[1].plot(np.sin(x))
    ax[1].plot(np.cos(x))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')
with plt.style.context('classic'):
    hist_and_lines()
# 将其中的‘classic’字符串替换成你想要的风格名称，就能在with管理区内使用风格，而不影响后面的绘图。
with plt.style.context('fivethirtyeight'):
    hist_and_lines()
# ggplot
with plt.style.context('ggplot'):
    hist_and_lines()
#bmh
with plt.style.context('bmh'):
    hist_and_lines()

# 灰度grayscale
with plt.style.context('grayscale'):
    hist_and_lines()
# seaborn
with plt.style.context('seaborn'):
    hist_and_lines()






















































































































