# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:15:59 2022

@author: zhaokai
"""
from operator import index
import numpy as np
import pandas as pd
""" 
4 选择数据
"""
df = pd.DataFrame(np.random.randint(0, 150,size=(1000,3)), columns=['python', 'en', 'math'])
# 列的获取
df["python"]
df.python 
df[['python', 'en']]
# 行的获取
df2 = pd.DataFrame(np.random.randint(0,150,size=(5,3)),index=list("ABCDE"),columns=["python","math","en"])
df2
df2.loc["A"]
df2.loc[["A","D"]]
df2.iloc[0]
df2.iloc[[0,3]]

# 具体数值的获取
df2["math"]["B"]
##  loc表示，先获取行，再获取列
df2.loc["B"]["math"]
df2.iloc[1,1]
df2.loc["B","math"]

# 切片
df2.loc["A":"C","math":]


""" 
布尔值的索引 boolean
"""
cond = df["python"]==149
df[cond]

cond1 = df2["python"]>130
cond2 = df2["math"]>130
cond = cond1 & cond2
df2[cond] 

"""
5. 数据筛选 

"""
## 赋值操作
df = pd.DataFrame(np.random.randint(0, 51,size=(20,3)), columns=['python', 'math', 'en'])
df
# 增加一列
df["物理"]=np.random.randint(0,51,size=20)
df 
# 将python 列的值增加10
df["python"] +=10
df 
# 将math 索引为2 的这个人的数据增加10
df["math"][2] +=10
df
# 将math 索引为4 5  的这个人的数据变成为100
df["math"][[4,5]] = 100
df
# 批量修改多个数据
df.loc[[1,2,3,4,5],["python","math"]] = 1024
df
## 如果shi条件：loc方式修改数据
cond = df["物理"]>=50
df.loc[cond] -=100
df[cond]


""" 
6. 数据集成

"""
# 方式1 concat
df2 = pd.DataFrame(np.random.randint(0,150,size=(5,4)),index=list("ABCDE"),columns=["python","math","en","物理"])
# np.concatenate((df,df2),axis=1)
# axis = 1 表示横向拼接 axis = 0 表示纵向拼接
pd.concat([df,df2],axis=0)

df3 = pd.DataFrame(np.random.randint(0, 51,size=(20,3)), columns=['java', 'niubi', 'haha'])
pd.concat([df,df3],axis=1)
pd.concat([df,df3],axis=0) #行列不一致  出现空值

# 方式2 insert
df.insert(loc=3,column="生物",value=np.random.randint(50,99,size=20))
df

# 需求：在python 列后面插入一列 ”niubi“
#1.获取列索引
index = list(df.columns).index("python") +1
df.insert(loc=index,column="niubi",value=np.random.randint(50,99,size=20))
df

# 方式3 join SQL 数据库风格的插入和合并--merge
df1 =pd.DataFrame(data = {"name":["softpo","dannie","brandon","ella"],"weight":[70,55,75,65]})
df2 =pd.DataFrame(data = {"name":["softpo","dannie","brandon","ella"],"height":[172,170,175,165]})
df3 =pd.DataFrame(data = {"名字":["softpo","dannie","brandon","ella"],"weight":[70,55,75,65]})
pd.merge(df1,df2,on="name") # 根据共同的属性 进行合并

pd.merge(df1,df3,left_on="name",right_on="名字") # 根据共同的属性 进行合并


df2 = pd.DataFrame(np.random.randint(0,150,size=(5,4)),index=list("ABCDE"),columns=["python","math","en","物理"])
df2.mean()
# 每个人的平均分
s = df2.mean(axis=1).round(2)

df5 = pd.DataFrame(s,columns=["平均分"])
df5

# 可以根据行索引合并  --适用于没共同的属性
pd.merge(df2,df5,left_index=True,right_index=True)

""" 
7. 数据清洗

"""
df = pd.DataFrame(data ={"color":["red","blue","red","green","red",None,"red"],"price":[60,20,60,40,60,np.NAN,60]})
df
# 重复数据过滤
df.duplicated() # 判断是否存在重复数据
df.drop_duplicates() # 删除重复数据

# 空数据过滤
df.isnull()
df.dropna()
df.fillna(1024)

# 指定行或列过滤
#!! 注意：这里没有修改原数据
df.drop(labels=["price"],axis=1) # 删除指定列

df.drop(labels=[0,1,2],axis=0) # 删除指定行

# inplace 会修改原数据
df.drop(labels=[0,1,2],axis=0,inplace=True) # 删除指定行
df

# 异常值的过滤
df2 = pd.DataFrame(data=np.random.randn(10000,3)) # 正态分布数值
df2
df2.mean()
df2.std()
# 3个标准差之内的数据  之外就是异常值
# 比较运算
cond = df2.abs() > 3*df2.std()
cond.sum()

cond_0 = cond[0]
df2[cond_0]

# 获取每一列的异常值
cond = df2.abs() > 3*df2.std()
cond_0 = cond[0]
cond_1 = cond[1]
cond_2 = cond[2]
# 逻辑运算
cond_ = cond_0 | cond_1 | cond_2
df2[cond_]

cond_=cond.any(axis=1) # 只要有一个为True就为True
df2[cond_]

""" 
8. 数据转换

"""
df = pd.DataFrame(data ={"color":["red","blue","red","green","red",None,"red"],"price":[60,20,60,40,60,np.NAN,60]})
df
# 重命名轴索引
df.rename(index=str,columns={"color":"颜色","price":"价格"})

# 替换值
df.replace(60,100) # 替换指定值
df.replace([100,20],2048) # 替换0、7为指定值
df.replace({0:512, np.NAN:998}) # 根据字典的键值对进行替换
df.replace({"color":"red"},"niubi") # 将color列中的red替换为-1024

# map series 
# 1.map 批量元素改变，series 专有
df["color"].map({"red":"RED","blue":"BLUE"})  # 字典的映射

# 利用函数进行替换
def convert(x):
    if x == "red":
        return "RED"
    if x=="blue":
        return "BLUE"
    if x=="green":
        return "GREEN"
    else:
        return x
df["color"].map(convert)

# 隐式函数映射
df["price"].map(lambda x:True if x>30 else False)


""" 
9. apply 元素改变，既支持series 也支持dataframe

"""
# 对一列数据进行了转换  并且插入到了原数据中

df = pd.DataFrame(np.random.randint(0,100,size=(30,3)),columns=["python","math","en"])
df
def convert(x):
    if x<60:
        return "不及格"
    elif x<80:
        return "中等"
    else: 
        return "优秀"
df["python"].apply(convert)

# 添加到原有的列表中
# 对一列数据进行了转换  并且插入到了原数据中
result = df["python"].apply(convert)
index = list(df.columns).index("python") +1
df.insert(loc=index,column="python_grade",value=result)
df

###  通过for 循环进行操作 
# 对整个dataframe进行操作 进行整体转换
del df["python_grade"]
df

def convert(x):
    if x<60:
        return "不及格"
    elif x<80:
        return "中等"
    else: 
        return "优秀"
for col in ["python","math","en"]:
    result = df[col].apply(convert)
    # 插入位置的索引 变量来表示
    index = list(df.columns).index(col) +1
    df.insert(loc=index,column=col+"_grade",value=result)
df
