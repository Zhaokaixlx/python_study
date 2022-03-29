# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:15:59 2022

@author: zhaokai
"""
from sys import prefix
import numpy as np
import pandas as pd
df = pd.read_excel(r"D:\pycharmcode\13\分数汇总.xlsx")
df.head()
# 保存时，需要指定编码
df.to_excel(r"D:\pycharmcode\13\分数汇总.xlsx",encoding="utf-8")

""" 
apply 应用
方法可以自定义  可以接受一个函数作为参数
"""
df = pd.DataFrame(np.random.randint(0, 10,size=(10,3)), index = list("abcdefhijk"),columns=['python', 'math', 'en'])
df 
df["python"] = df["python"].apply(lambda x : x+100)
df
df["python"] = df["python"].map(lambda x : x-100)
df 
# 所有数据都发生变化
# applymap 对所有数据都发生变化 对整个dataframe进行全部处理
def convert(x):
    if x<5:
        return x+100
    else:
        return x-100
df.applymap(convert)

""" 
transform 变换  数据转换
"""
df = pd.DataFrame(np.random.randint(0, 10,size=(10,3)), index = list("abcdefhijk"),columns=['python', 'math', 'en'])
df["python"] = df["python"].transform(lambda x : x+100)
df
# 对一列进行不同的操作 apply transform
df["python"].transform([np.sqrt,np.exp])
df["python"].transform([np.exp,np.sqrt,convert])

# 对多列进行不同的操作 
df.transform({'python':[np.sqrt,np.exp],'math':[np.exp,np.sqrt,convert]})


""" 
重排 随机抽样 哑变量

"""
df = pd.DataFrame(np.random.randint(0, 10,size=(10,3)), index = list("abcdefhijk"),columns=['python', 'math', 'en'])

index = np.random.permutation(df.index)
index = np.random.permutation(10)
print(index)
df.take(index)

# 随机抽样  可以重复
index = np.random.randint(0,10,size=5)
df.take(index)

# 哑变量  独热编码 0表示无 1表示有
df = pd.DataFrame({"key":["b","b","a","c","a","b"]})
df
pd.get_dummies(df,prefix="",prefix_sep="") 



""" 
数据重塑

"""
df = pd.DataFrame(np.random.randint(0, 100,size=(10,3)), index = list("abcdefhijk"),columns=['python', 'math', 'en'])
df
df.T  # 转置

## 多层索引
df2 = pd.DataFrame(data = np.random.randint(0,100,size = (20,3)),
                   index = pd.MultiIndex.from_product([list('ABCDEFHIJK'),['期中','期末']]),#多层索引
                   columns=['Python','Tensorflow','Keras'])
df2
df3 = pd.DataFrame(data = np.random.randint(0,100,size = (10,6)),index =list("ABCDEFHIJK"),columns=pd.MultiIndex.from_product([["python","math","en"],["期中","期末"]]))
df3

# 行索引 变成为列索引  结构改变
# 默认情况下  最里层调整
df2.unstack()
df2.unstack(level = 0)
df2.unstack(level=1)
# 列索引变成行索引
df3.unstack(level = -1)
df3.unstack(level = 0)
df3

# 多层索引的运算
df.sum()
df.sum(axis=1)

# 期中期末消失 每个人的期中期末的总分
df2.sum(level = 0)
df2.mean(level = 0)
# 期中期末的总分
df2.sum(level = 1) 
df2.mean(level = 1)
 
# df3是多层索引，可以直接使用[]，根据层级关系取数据
df3
df3["python","期中"]
df3["python","期中"] ["A"]


"""  
数学和统计方法

"""
# 简单统计指标
df = pd.DataFrame(np.random.randint(0, 100,size=(10,3)), index = list("abcdefhijk"),columns=['python', 'math', 'en'])
def convert(x):
    if x>80:
        return np.NAN
    else:
        return x
df["python"] = df["python"].apply(convert)
df["math"] = df["math"].map(convert)
df["en"] = df["en"].transform(convert)
df
# 返回有多少个非空数据的个数
df.count()

df = pd.DataFrame(np.random.randint(0, 100,size=(10,3)), index = list("abcdefhijk"),columns=['python', 'math', 'en'])
# 中位数
df.median()
df
# 百分位数
df.quantile(0.5)
df.quantile([0.5,0.75,0.9])

# 最大值
df.max()


"""  
索引标签  位置获取

"""
# 索引位置
# 计算最大值的位置
df["python"].argmax()
# 计算最小值的位置
df["python"].argmin()

## 返回索引标签
df.idxmax()
df.idxmin()
df["python"].loc["a"]
df["python"].iloc[0]
df.min()

""" 
更多的统计指标

"""
# 统计元素出现的次数
df["python"].value_counts()

# 去重
df["python"].unique()
# 累加 累乘
df.cumsum()
df.cumprod()
# 标准差
df.std()
# 方差
df.var()
# 累计最小值 累计最大值
df.cummin()
df.cummax()

# 计算差分 和上一行相减
df.diff()
#计算百分比的变化
df.pct_change()


""" 
高级的统计指标

"""
df.cov() # 属性的协方差
df["python"].cov(["math"]) # python 和math 的协方差

df["python"].corr(df["math"]) # 属性的相关系数

# 数据的排序
import numpy as np
import pandas as pd

df1 = pd.DataFrame(data=np.random.randint(0,30,size=(30,3)),
                  columns=["python","keras","pytorch"])
df1

# 1.索引列名进行排序
df1.sort_index(axis=0,ascending=False) #按照索引进行排序 降序
df1.sort_index(axis=1,ascending=False) # 按照索引进行排序 降序

# 2.属性值排序
df1.sort_values( by=["python"])  # 按照python的属性值进行排序
df1.sort_values( by=["python","keras"])  # 先按照python的属性值进行排序
                                         # 再按照keras排序
# # 3、返回属性n大或者n小的值
df1.nlargest(10,columns='keras') # 根据属性Keras排序,返回最大10个数据
df1.nsmallest(5,columns='python') # 根据属性Python排序，返回最小5个数据                                       



""" 
分箱操作  --离散化

"""
df = pd.DataFrame(data = np.random.randint(0,150,size = (100,3)),
                  columns=['Python','Tensorflow','Keras'])
df                                         

# 1、等宽分箱
pd.cut(df.Python,bins = 3)
# 指定宽度封箱
pd.cut(df.Keras,# 分箱的数据
       bins=[0,60,90,120,150],# 分箱的断点
       right=False,# 左闭右开
       labels=["不及格","中等","良好","优秀"] # 分箱后的分类
       )

# 2. 等频分箱
t = pd.qcut(df.Python,q =4, # 4等分
       labels=["不及格","中等","良好","优秀"] # 分箱后的分类
       )
t.value_counts()

# 用函数来实现
def convert(x):
    if x<60:
        return "不及格"
    elif x<90:
        return "中等"
    elif x<120:
        return "良好"
    else:
        return "优秀"
df.Keras.map(convert) 


""" 
分组聚合
groupby 过程拆解
分组层层拆解
超级炫酷
"""
import numpy as np
import pandas as pd
# 准备数据
df = pd.DataFrame(data = {'sex':np.random.randint(0,2,size = 300), # 0男，1女
                          'class':np.random.randint(1,9,size = 300),#1~8八个班
                          'Python':np.random.randint(0,151,size = 300),#Python成绩
                          'Keras':np.random.randint(0,151,size =300),#Keras成绩
                          'Tensorflow':np.random.randint(0,151,size=300),
                          'Java':np.random.randint(0,151,size = 300),
                          'C++':np.random.randint(0,151,size = 300)})
df['sex'] = df['sex'].map({0:'男',1:'女'}) # 将0，1映射成男女
df

# 根据性别分组  并且求其平均值
df.groupby('sex').mean().round(2)


# 根据性别分组  并且求其有多少个数据
df.groupby("sex").size()

# 先根据性别分组，之后根据班级进行分组  并且求其有多少个数据
df.groupby(by=["sex","class"]).size()

# 先根据性别分组，之后根据班级进行分组  并且求“python” “Java” 这两列的最大值
df.groupby(by=["sex","class"])[["Python","Java"]].max()
# 用unstack 行变列
df.groupby(by=["class","sex"])[["Python","Java"]].max().unstack()

""" 
分组聚合 ---apply transform
  优势在于  可以直接在里面写函数
"""
# 分组后调用apply，transform封装单一函数计算
# 返回分组结果
df.groupby(by = ['class','sex'])[['Python','Keras']].apply(np.mean).round(1)

df.groupby(by = ['class','sex'])[['Python','Keras']].transform(np.mean).round(1)

df.head()

""" 
分组聚合 agg
优势： 可以针对不同的列  采用不同的操作

"""
# 多种统计汇总操作
df.groupby(by = ['class','sex'])[['Tensorflow','Keras']].agg([np.max,np.min,pd.Series.count])

# 分组后不同属性应用多种不同统计汇总
df.groupby(by = ['class','sex'])[['Python','Keras']].agg({'Python':[('最大值',np.max),('最小值',np.min)],
                                                          'Keras':[('计数',pd.Series.count),('中位数',np.median)]})


""" 
透视表  - 也是一种分组聚合运算

"""
def count(x):
    return len(x)
df.pivot_table(values=['Python','Keras','Tensorflow'],# 要透视分组的值
               index=['class','sex'], # 分组透视指标,by = ['class','sex']
               aggfunc={'Python':[('最大值',np.max)], # 聚合运算
                        'Keras':[('最小值',np.min),('中位数',np.median)],
                        'Tensorflow':[('最小值',np.min),('平均值',np.mean),('计数',count)]})













