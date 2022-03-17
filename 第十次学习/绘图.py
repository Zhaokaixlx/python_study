# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:24:22 2022

@author: Administrator
"""

"""
散点图
scatter(x,y,lolor,s)
 x:自变量 y:因变量
 color：散点的颜色
 s:散点的大小
 散点的形状：marker = "x"
 .(散点)  ,（正方形） o（圆） ^（上三角形）
 <（左三角形）   >（右三角形）
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
plt.scatter(x, y, color="k",s=50,marker=">")
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


"""
绘制折线图
plot(x,y,color)
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False



x = list(range(1,11))
y = [pow(i,2) for i in x ]
# 定义折线图图的标题
plt.title("赵凯的表啊",fontsize = 15,color = "black")
# 定义折线图的坐标轴
plt.xlabel("x轴呀", fontsize = 15,color = "r")
plt.ylabel("y轴呀", fontsize = 15,color = "g")
# 限制坐标轴
plt.xlim((0, 15))
plt.ylim((0, 150))
# 自定义刻度
my_x_ticks = np.arange(0,20,1)
my_y_ticks = np.arange(0,120,10)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
# 绘图展示
plt.plot(x, y, color="c",linestyle =":")
plt.show()


"""
线型和颜色
1.颜色
r:red() 红色 b:blue 蓝色
g:green 绿色 c:cyan 青色
m:magenta 洋红 y:yellow 黄色
k:black 黑色 w:white 白色
2.线型(linestyle)
-粗线  --虚线  -.点画线  
：点   
"""

# 转换为日期格式的数据
# df["时间"] = pd.to_datetime(df["时间"])

# 线宽 linewidth = 10或者任意大小数字



"""
绘制直方图
hist(x,color,bins,cumulative)
x: 需要绘制的向量
color:直方图颜色
bins:设置直方图的分组个数
cumulative:是否设置累计计数
准备数据 df
"""
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False
df = pd.DataFrame(np.arange(600).reshape(100,6),columns =["a","b","c","d","e","f"] )
print(df)

# 标题
plt.title("a", fontsize=25,color="c")
# 坐标
plt.xlabel("统计",fontsize=25,color="r" )
plt.ylabel("频数",fontsize=25,color="k" )

# 绘制直方图
plt.hist(df["a"], color="c")
plt.hist(df["a"], color="c",cumulative=True)
plt.hist(df["a"], color="c",bins = 20)


"""
绘制柱状图
bar(left,height,width,color)
left: x
height:y
width:控制柱形图的柱宽
color:制柱形图的颜色
"""
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False
df = pd.DataFrame(np.arange(600).reshape(100,6),columns =["a","b","c","d","e","f"] )
print(df)

# 定义变量
name = ["zhaokai","zhaoli","mama","baba","laolao","taizi","cisa"]
gra = [3.1 ,2.2 ,2.8,2.9,3.0,1.8,1.7]

# 绘制柱状图
index = range(0,len(name))
plt.xticks(index,name)
# 显示成绩  "%.2f"%y 对于y保留两位小数 ha 水平位置 va 垂直位置
for x,y in zip(index,gra):
    plt.text(x,y,"%.2f"%y,ha = "center",va ="bottom")
plt.bar(index, gra, width=0.8,color = "b")



"""
绘制堆积柱状图

"""
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False
df = pd.DataFrame(np.arange(600).reshape(100,6),columns =["a","b","c","d","e","f"] )
print(df)

name = ["zhaokai","zhaoli","mama","baba","laolao","taizi","cisa"]
gra = [3.1 ,2.2 ,2.8,2.9,3.0,1.8,1.7]
gra1=[2.9,2.8,3.5,3.6,3.3,1.9,2.7]
gra2=[3.3 ,2.9 ,2.9,2.1,3.9,1.9,1.5]
gra3= [4.2 ,3.0 ,3.8,3.9,3.3,2.8,2.7]
# 替换坐标轴
index = range(0,len(name))
plt.xticks(index,name)
# 绘图
plt.bar(index, gra, width=0.8,color = "#0099ff")
plt.bar(index, gra1,bottom = gra, width=0.8,color = "b")
plt.bar(index, gra2,bottom = np.array(gra)+np.array(gra1), width=0.8,color = "r")
plt.bar(index, gra3,bottom = np.array(gra)+np.array(gra1)+np.array(gra2), width=0.8,color = "y")

# legend(图标)
plt.legend(["gra","gra1","gra2","gra3"])


"""
绘制横向柱状图

"""
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False

# 定义变量
name = ["zhaokai","zhaoli","mama","baba","laolao","taizi","cisa"]
gra = [3.1 ,2.2 ,2.8,2.9,3.0,1.8,1.7]

index = range(len(name))
plt.yticks(index,name)

# 绘制横向柱形图
plt.barh(index, gra, color="r")

# 绘制横向堆积柱形图

name = ["zhaokai","zhaoli","mama","baba","laolao","taizi","cisa"]
gra = [3.1 ,2.2 ,2.8,2.9,3.0,1.8,1.7]
gra1=[2.9,2.8,3.5,3.6,3.3,1.9,2.7]
gra2=[3.3 ,2.9 ,2.9,2.1,3.9,1.9,1.5]
gra3= [4.2 ,3.0 ,3.8,3.9,3.3,2.8,2.7]

index = range(len(name))
plt.yticks(index,name)
my_x_ticks = np.arange(0,11,1)
plt.xticks(my_x_ticks)


plt.barh(index, gra, color="r")
plt.barh(index, gra1,left = gra,color = "b")
plt.barh(index, gra2,left = np.array(gra)+np.array(gra1),color="y")
plt.barh(index, gra3,left = np.array(gra1)+np.array(gra2),color="c")

# 图例
plt.legend(["gra","gra1","gra2","gra3"])


"""
绘制双向横向柱状图

"""
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False

# 定义变量
name = ["zhaokai","zhaoli","mama","baba","laolao","taizi","cisa"]
gra = [3.1 ,2.2 ,2.8,2.9,3.0,1.8,1.7]
gra1=[2.9,2.8,3.5,3.6,3.3,1.9,2.7]

# 替换坐标轴
index = range(len(name))
plt.yticks(index,name)

# 绘制双向柱形图
plt.barh(index,gra,color = "r")
plt.barh(index,-np.array(gra1),color = "b")
plt.legend(["gra","gra1"])


"""
绘制饼图
pie(x,labels,colors,explode,aotupcct)
x:进行绘制图形的序列
labels:饼图的各部分标签序列
colors；饼图各部分的颜色，RGB
explode:需要突出的块状序列
autopct:饼图占比的显示格式

"""
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False

# 定义变量
num = [20,30,40,50]
grade = ["一年级","二年级","三年级","四年级"]
colors=["orange","purple","red","blue"]
explode = (0.2,0,0,0)
plt.title("各年级人数", fontsize=20)
plt.axis("equal") # 保证是一个圆，而不是椭圆
plt.pie(num,labels=grade,colors = colors,explode=explode,autopct="%.2f%%",shadow =False,pctdistance=0.6,startangle=0)

# 修改图例坐标的方式
plt.legend(["一年级","二年级","三年级","四年级"])
plt.legend(grade, bbox_to_anchor = [0.1,1])

"""
绘制箱线图

"""
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False


df = pd.read_excel(r"D:\pycharmcode\第十次学习\data1.xlsx")


# 使用matplotlib绘制
plt.boxplot(x=df.values,labels=df.columns,whis=1.5)
 
plt.show()

# 使用pandas 绘制
df.boxplot()
plt.show()


"""
图像弹窗


"""
import matplotlib.pyplot as plt


y = x

plt.plot(x,y)
plt.show()
 



"""
多图的叠加


"""
import matplotlib.pyplot as plt
x = list(range(1,11))
y = [9,6,8,22,23,24,56,89,6,30]
z = [10,16,28,12,3,4,96,60,50,30]


plt.plot(x,y)
plt.plot(x,z)
plt.legend(["y","z"])
plt.show()

"""
子图的绘制
subplot()

"""
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np

# 画板的设置
fig = plt.figure(figsize=(6,3))

# 数据生成
x = np.linspace(0, 10,100)

# 定义y
y1 = np.sin(x)
y2 = np.cos(x)

# 子图的绘制
plt.subplot(2,2,1)
plt.plot(x,y1,"r--")
plt.title("f(x)=sin(x)")


plt.subplot(2,2,4)
plt.plot(x,y2,"r--")
plt.title("f(x)=cos(x)")

####  subplots 函数 也可以
# 有两个返回值 1. 图像[fig] 2.图像的内容[ax]

fig,ax = plt.subplots(2, 2,figsize=(8,6))
ax[0][0].plot(x,y1)
ax[0][0].set_title("f(x)=sin(x)")


ax[1][1].plot(x,y2)
ax[1][1].set_title("f(x)=cos(x)")

# 解决标题遮挡
plt.tight_layout()


"""
绘制心形图
x = 16*np.sin(t)**3
y = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
t:必须是一个完整的正弦、余弦曲线
""" 
import matplotlib.pyplot as plt
import numpy as np

# 完整正弦波
t = np.linspace(0, 7,101)
#y = np.sin(t)
#plt.plot(t,y)
x = 16*np.sin(t)**3
y = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)

# 绘制心形图
plt.plot(x,y,linewidth=3,color="pink")
plt.xlim(-20, 20)
plt.ylim(-20, 15)


plt.title("LOVE",fontsize=20)
plt.fill_between(x, y, facecolor = "pink")
plt.show()

"""
绘图实战
""" 

import matplotlib.pyplot as plt
import numpy as np

# 设置中文
mpl.rcParams["font.sans-serif"]=["simhei"]
# 解决图像保存中负号：“-” 显示为方框的问题
mpl.rcParams["axes.unicode_minus"] = False

name = ["zhaokai","zhaoli","mama","baba","laolao","taizi","cisa"]
gra = [3.1 ,2.2 ,2.8,2.9,3.0,1.8,1.7]


means = np.mean(gra)
z = np.full((1,len(name)),means).ravel()



# 柱形图的绘制
index = range(len(name))
plt.xticks(index,name)

for x,y in zip(index,gra):
    plt.text(x, y, "%.2f"%y, ha="center",va="bottom")
plt.bar(index, gra,width=0.5)  

# 绘制平均数线
plt.plot(index,z,"-",color="y")

plt.plot(index, gra,color = "r")

# 加个图例
plt.legend(["mean:{}".format(means),"single-point"])

plt.show()
















