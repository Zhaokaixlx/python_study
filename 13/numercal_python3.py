# pandas
"""
1.数据结构

"""
#一维数据
from operator import index
import pandas as pd
import numpy as np
"""
一维数组series和之前numpy(自然索引)不同，series是一维数组，
可以通过索引访问元素，一一对应。
"""
s = pd.Series(data=[1,2,3,4,5],index=['a','b','c','d','e'])

# 二维结构 和excel一样
pd.DataFrame(data=np.random.randint(0,150,size=(5,3)),columns=['python','en','math'],index=list('abcde'),dtype=np.int32)
print(list("abcde"))

# excel类似
df = pd.DataFrame(data={"python":np.random.randint(100,150,size=5),
                   "en":np.random.randint(0,150,size=5),
                   "math":np.random.randint(0,150,size=5)},index=np.arange(1,6))

# index自增
df.sort_index(ascending=False)

"""
2. 数据查看

"""
df = pd.DataFrame(data=np.random.randint(0,151,size=(150,3)),columns=['python','en','math'],index=None,dtype=np.int32)

# 查看其属性、概览和统计信息
df.head(10)  # 查看前10行,默认5个
df.tail(10)  # 查看后10行,默认5个
# 查看形状，行数和列数
df.shape
# 查看数据类型
df.dtypes
#行索引
df.index
# 列索引
df.columns
# 对象值 二维ndaray数组
df.values
# 查看数值型列的汇总统计信息，计数、均值、标准差、最小值、四分位数、最大值
df.describe()
# 查看你索引、数据类型、非空计数和内存信息
df.info()

""" 
3. 数据的输入和输出

"""
# 创建一个数据  薪资情况
df = pd.DataFrame(data=np.random.randint(0,50,size=(50,5)),columns=['it','化工','生物',"教师","士兵"],index=None,dtype=np.int32)
print(df)

# 保存文件
df.to_csv(r"D:\pycharmcode\13\salary.csv",sep=",",header=True,index=False)

# 加载文件
pd.read_csv(r"D:\pycharmcode\13\salary.csv",sep=",",header=0,index_col=None)

df1 = pd.DataFrame(data = np.random.randint(0,50,size = [50,5]), # 薪资情况
               columns=['IT','化工','生物','教师','士兵'])
df2 = pd.DataFrame(data = np.random.randint(0,50,size = [150,3]),# 计算机科目的考试成绩
                   columns=['Python','Tensorflow','Keras'])


# 保存到当前路径下，文件命名是：salary.xls
df1.to_excel('./salary.xlsx',
            sheet_name = 'salary',# Excel中工作表的名字
            header = True,# 是否保存列索引
            index = False) # 是否保存行索引，保存行索引

df2.to_excel('./salary.xlsx',
            sheet_name = 'kaoshi',# Excel中工作表的名字
            header = True,# 是否保存列索引
            index = False) # 是否保存行索引，保存行索引

pd.read_excel('./salary.xlsx',
              sheet_name='salary',# 读取哪一个Excel中工作表，默认第一个
              header = 0,# 使用第一行数据作为列索引
              names = list('ABCDE'),# 替换列索引
              index_col = 3)# 指定行索引，B作为行索引
# 一个Excel文件中保存多个工作表
with pd.ExcelWriter('./data.xlsx') as writer:
    df1.to_excel(writer,sheet_name='salary',index = False)
    df2.to_excel(writer,sheet_name='score',index = False)
pd.read_excel('./data.xlsx',
              sheet_name='score') # 读取Excel中指定名字的工作表

""" 
4. 数据选择

"""







