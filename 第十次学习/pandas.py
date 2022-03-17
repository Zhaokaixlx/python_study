# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 21:00:20 2022

@author: Administrator
"""
"""
pandas:
    数据分析工具
    内置了很多函数
    基本数据类型：1.series  一列数据
                 2.dataframe 相当于excel中的sheet
                 
"""


"""
series(系列)
可以看做竖起来的list

"""
import pandas as pd

# 生成第一个series[按照默认的index]
s_1 = pd.Series([1,2,3,4,5])
print(s_1)

# 自定义index
s_2 = pd.Series([1,2,3,4,5],index=["a","b","c","d","e"])
print(s_2)

s_3 = pd.Series(["zhao kai","zhao li","ma ma","ba ba","lao lao"],index=["a","b","c","d","e"])
print(s_3)

# series 的一些属性
s_1.index
s_3.values



"""
对series进行操作

"""
# 查 1.标签访问
s_3["a"]
s_3["a":"d"]
s_3[["a","d"]]
# 2.索引访问
s_2[0]
s_3[4]
s_3[0:3]
s_3[[0,2]]

# 增
s_4 = pd.Series(["999"],index=["f"])
s_2 = s_2.append(s_4)

# 删
s_2 = s_2.drop("f")

print(s_2)

# 判断某个值是否在Series里面
print("yiyi" != s_3.values)

# 改
s_3[4] ="po po"
print(s_3)

# 创建Series
dic_1 = {"name":"zhao kai","gender":"male","hometown":"shanxi lvliang"}
s_5 = pd.Series(dic_1)
print(s_5)

# 重置索引
s_5.index = range(1,len(s_5)+1)




"""
DataFrame(数据框)
相当于excel中的sheet
或者可以理解为多个series的拼接
"""
df_1 = pd.DataFrame({"name":["zhao kai","zhao li","ma ma"],"age":[26,22,40],"income":[999999,8888,77777]}
                    ,index=["p1","p2","p3"])
print(df_1)

"""
dataframe的属性
"""
# 行索引
df_1.index
# 列名
df_1.columns
# 值
df_1.values

"""
dataframe的操作
"""
# 修改
print(df_1.columns)
df_1.columns = range(0,len(df_1.columns))
print(df_1.columns)

# 精准修改
df_1.rename(columns = {"age":"年龄","name":"姓名"},inplace = True)
print(df_1.columns)

df_1.index = range(0,len(df_1.index))
print(df_1.index)


"""
增加一列
"""
df_1["pay"] = [20,30,40]
df_1.insert(0,"22",df_1.pop("pay"))
df_1.insert(0,"23",[2,20,222])

"""
增加一行
"""
df_1.loc["p4",["name","age","income"]]=["tai zi",20,20000]

"""
访问Dataframe
"""
# 列
df_1.name
df_1.age

df_1[["age","income"]]
df_1[[0,2]]

# 访问行
df_1[0:2]
df_1.loc[["p1","p4"]]

# 访问某个值
df_1.loc["p1","name"]


# 删除  直接在源数据上删除
del df_1["age"]

# 删除列
data = df_1.drop("age",axis = 1,inplace = False)
# 删除行
data = df_1.drop("p4",axis = 0,inplace = False)
print(data)


"""
Dataframe 查询的三种方法
loc()
iloc()
ix()
"""
# loc() df_1.loc[x,y]
#标签索引
# 打印某个值
df_1.loc["p3","age"]

# 打印某列
df_1.loc[:,"age"]
# 打印某几列
df_1.loc["p3":,:"age"]

df_1.loc[:,["age","income"]]

# 打印某行
df_1.loc["p3",:]

# 打印某几行
df_1.loc["p2":,:]

# iloc() df_1.iloc[x,y] 
# 位置索引
df_1.iloc[0,0]
# 打印某列
df_1.iloc[:,2]
# 打印某几列
df_1.iloc[:,[0,2]]
# 打印某行
df_1.iloc[1,:]

# ix() df_1.ix[x,y]
# 混合索引
 
# 访问某些元素
df_1.ix["p1":"p3",[1,2]]

df_1.ix[:,[2]]


"""
dataframe的操作
用df_1
"""

# 根据年龄这一列排序
# 升序
df_1 = pd.DataFrame({"name":["zhao kai","zhao li","ma ma"],"age":[26,22,40],"income":[999999,8888,77777]},index=["p1","p2","p3"])
df_1.sort_values(by= "age")
# 降序
df_1.sort_values(by= "age",ascending = False)

# 值替换
df_1["age"].replace(["40","26","22"],["40岁","26岁","22岁"])
df_1["age"].replace(["40","26"],["40岁","26岁"])
df_1["age"] = df_1["age"].replace(["40","26","22"],["40岁","26岁","22岁"])

# 重新排列数据中的列
cols = ["income","age","name"]
df_1 =df_1.loc[:,cols]

"""
数据的导入 和 导出

"""
# 读取xxxx.csv文件
import pandas as pd
df = pd.read_csv(r"xxx\xxx\xxx\文件名.csv",engine = "python",encoding = "utf-8")
# 取消将第一行数据作为列名的问题
df = pd.read_csv(r"xxx\xxx\xxx\文件名.csv",engine = "python",encoding = "utf-8",header = None)

# 读取xxxx.excel文件
df1= pd.read_excel(r"xxx\xxx\xxx\文件名.xlsx",engine = "python",encoding = "utf-8")
# 取消将第一行数据作为列名的问题
df1 = pd.read_excel(r"xxx\xxx\xxx\文件名.xlsx",engine = "python",encoding = "utf-8",header = None)
  
# 读取xxxx.txt文件 加分隔符sep = ","
df1= pd.read_table(r"xxx\xxx\xxx\文件名.txt",engine = "python",encoding = "utf-8",sep = ",")

# 取消将第一行数据作为列名的问题
df1= pd.read_table(r"xxx\xxx\xxx\文件名.txt",engine = "python",encoding = "utf-8",sep = ",",header = None)


"""
文件的导出

"""
df = pd.to_csv(r"xxx\xxx\xxx\文件名.csv",engine = "python",encoding = "utf-8",index=False,header = True)

df = pd.to_excel(r"xxx\xxx\xxx\文件名.xlsx",engine = "python",encoding = "utf-8",index=False,header = True)



"""
缺失值的处理方式
1.数据补齐 2.删除对应的数据行 3.不处理

"""

"""
1.进行逻辑判断，判定空值所在的位置
"""

na = df_1.isnull()

# 2.找出空值所在的行数据
df_1[na.any(axis = 1)]

# 3.找出空值所在的列数据(取出行数据)
df_1[na[["name"]].any(axis = 1)]
df_1[na[["age"]].any(axis = 1)]
df_1[na[["income"]].any(axis = 1)]
df_1[na[["name","income"]].any(axis = 1)]


# 填充缺失值
df_2 = df_1.fillna("99999")


# 删除缺失值 --删除了整行数据
df_3= df_1.dropna()


"""
重复值的处理方式

"""
# encoding = "gbk"
df = pd.read_csv(r"xxx\xxx\xxx\文件名.csv",engine = "python",encoding = "gbk")

# 1.找出重复值的位置
result1 = df_1.duplicated()

# 根据列名来判断重复
result2 = df_1.duplicated("name")
print(result2)
# 根据某些列
result2 = df_1.duplicated(["name","income"])

# 2.提取重复的行  仅提取df_1.duplicated()中的Ture的

df_1[result1]

# 3.完全重复的删除
new_df_1 = df_1.drop_duplicates()

# 4.部分重复的删除
new_df_2 = df_1.drop_duplicates(["name","income"])


"""
slice() 函数
字段截取函数，作用对象是字符串

"""
df = pd.DataFrame({"name":["zhao kai","zhao li","ma ma"],"age":[26,22,40],"income":[99999,8888,77777],"id":[321282198110275214,510283198111268631,230227198601300026]})

# 将id 转化为字符串
df["id"] = df["id"].astype(str)

# 提取前六位地址码 0-6
area = df["id"].str.slice(0,6)

# 提取出生日期码 6-14
birthday = df["id"].str.slice(6,14)

# 提取顺序码 14-17
ranking = df["id"].str.slice(14,17)

# 提取唯一校验码
only = df["id"].str.slice(17,18)

# 将信息添加到数据框
df["area"]=area
df["birthday"]=birthday
df["ranking"]=ranking
df["only"]=only


"""
数据抽取
根据一定条件，抽取数据

"""
# 比较运算：包含 大于、小于等运算
# 实质就在于 逻辑判断+取数
# 抽取收入》10000的人
df[df["income"]>10000]

# 抽取收入在1000-80000之间的人数
df[df["income"].between(1000,80000)]


"""
字符匹配
"""
# na = Ture 表示空值也打印出来；False 为空值不打印
df[df["name"].str.contains("zhao kai",na=False)]

"""
逻辑运算：&(且)、| （和）
"""
# 取出年龄在20-30之间,收入大于10000的人
#➕() 
df[(df["age"].between(20,30)) & (df["income"]>10000)]


"""
数据框的合并
concat(df1,df2,df3,......)

"""
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.arange(12).reshape(3,4))
df2 = 2*df1

# 竖向合并
new_df1 = pd.concat([df1,df2])
# 横向合并
new_df2 = pd.concat([df1,df2],axis = 1)


# join 参数 inner:表示交集 outer:表示并集
df3 = pd.DataFrame(np.arange(12).reshape(3,4),index=["a","b",2])
new_df3 = pd.concat([df1,df3],axis = 1,join="inner")
new_df3 = pd.concat([df1,df3],axis = 1,join="outer")


"""
使用 + 进行拼接 列
"""
df.columns
num  = df["area"] + df["birthday"] \
+ df["ranking"] +df["only"]

"""
merge 函数
拼接两个数据框
准备了 df 和 df1
"""
df = pd.DataFrame({"name":["zhao kai","zhao li","zuo bian"],"age":[26,22,40],"income":[99999,8888,77777],"id":[321282198110275214,510283198111268631,230227198601300026]})
df1 = pd.DataFrame({"name":["zhao kai","zhao li","youbian"],"age":[26,22,40],"income":[99999,8888,77777],"id":[321282198110275214,510283198111268631,230227198601300026]})

# 根据name将两个数据框连接起来，删除各自数据框独有的信息
df2 = pd.merge(df,df1,left_on = "name",right_on = "name")
# 根据name将两个数据框连接起来，保留左边数据框独有的信息
df3 = pd.merge(df,df1,left_on = "name",right_on = "name",how = "left")
# 根据name将两个数据框连接起来，保留右边数据框独有的信息
df4 = pd.merge(df,df1,left_on = "name",right_on = "name",how = "right")
# 根据name将两个数据框连接起来，保留所有数据框的信息
df5 = pd.merge(df,df1,left_on = "name",right_on = "name",how = "outer")

"""
数据框的计算

准备了 df
"""
df["应该收入"] = df["age"]*df["income"]
df.应该收入 = df["age"]*df["income"]
df.age*2


"""
随机抽样
1.按照个数抽样
2.按照比例抽样
"""
import pandas as pd
import numpy as np
df = pd.DataFrame(np.arange(600).reshape(100,6),columns =["a","b","c","d","e","f"] )
print(df)

# 设置随机种子
np.random.seed(seed = 2)
# 1.按照个数抽样  不放回
df.sample(10)
# 1.按照个数抽样  放回
df.sample(10,replace = True)


# 2.按照比例抽样 不放回
df.sample(frac = 0.2)
# 2.按照比例抽样 放回
df.sample(frac = 0.2,replace = True)

"""
数据的标准化
1.0~1标准化：也称离差标准化，它是对原始数据进行线性变换，使结果落到【0，1】区间内
Z = (x-min)/(max-min)
2.z标准化：数据均值为0，标准差为1.
Z = (x-mean)/std
round(xxx,2) 保留两位小数
标准化原因：消除量纲的影响
准备 df 数据
"""
df = pd.DataFrame(np.arange(1,601).reshape(100,6),columns =["山西","陕西","内蒙古","北京","广州","上海"] )
# 0~1标准化
df["山西数据标准化"] =round( (df.山西 - df.山西.min())/(df.山西.max() - df.山西.min()),2)
df["陕西数据标准化"] =round( (df.陕西 - df.陕西.min())/(df.陕西.max() - df.陕西.min()),2)

# 分析相关性
df["山西数据标准化"].corr(df["陕西数据标准化"])

# Z标准化
df["山西数据Z标准化"] =round( (df.山西 - df.山西.mean())/(df.山西.std()),2)
df["陕西数据Z标准化"] =round( (df.陕西 - df.陕西.mean())/(df.陕西.std()),2)

# 分析相关性
df["山西数据Z标准化"].corr(df["陕西数据Z标准化"])

"""
数据分组
cut()函数
cut(series,bins,right = Ture,labels = null)
series:需要分组的数据  一列
bins：分组的划分数组 【列表】
right：分组的时候，右边是否包括。默认闭区间
labels：分组的自定义标签
准备 df 数据
"""
# 分组
bins = [min(df.山西)-1,200,300,400,500,max(df.山西)+1]
df["山西分组"] = pd.cut(df.山西,bins)
df["山西分组"] = pd.cut(df.山西,bins,right =False)
# 自定义标签
label = ["200以下","200-300","300-400","400-500","500以上"]
df["山西分组"] = pd.cut(df.山西,bins,right =False,labels = label)


"""
散点图
scatter(x,y,lolor,s)
 x:自变量 y:因变量
 color：散点的颜色
 s:散点的大小
"""
import matplotlib.pyplot as plt
x = list(range(1,11))
y = [pow(i,2) for i in x ]
# 定义散点图的标题
plt.title("zhao kai",fontsize = 15,color = "black")
# 定义散点图的坐标轴
plt.xlabel("x", fontsize = 15,color = "r")
plt.ylabel("y", fontsize = 15,color = "g")
# 绘图展示
plt.scatter(x, y, color="y",s=10)
plt.show()

###### 解决中文的问题
import matplotlib.pyplot as plt
from pylab import mpl
# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False



x = list(range(1,11))
y = [pow(i,2) for i in x ]
# 定义散点图的标题
plt.title("赵凯的表啊",fontsize = 15,color = "black")
# 定义散点图的坐标轴
plt.xlabel("x轴呀", fontsize = 15,color = "r")
plt.ylabel("y轴呀", fontsize = 15,color = "g")
# 绘图展示
plt.scatter(x, y, color="y",s=10)
plt.show()

"""
自定义坐标轴
"""
import matplotlib.pyplot as plt
from pylab import mpl
# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False

grade = ["一年级","二年级","三年级","硕士","博士"]
gpa =  [1.1,2.1,3.3,5.5,9.8]
# 定义散点图的标题
plt.title("各年级GPA",fontsize = 15,color = "black")
# 定义散点图的坐标轴
plt.xlabel("年级", fontsize = 15,color = "r")
plt.ylabel("各年级平均GPA", fontsize = 15,color = "g")
index = list(range(1,6))
# 对于x轴刻度的设置
plt.xticks(index,grade,fontsize=15,rotation=30)
# 绘图展示
plt.scatter(index,gpa, color="y",s=10)
plt.show()






