# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:22:01 2022

@author: Administrator
"""
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1、线形图
df1 = pd.DataFrame(data = np.random.randn(1000,4),
                  index = pd.date_range(start = '27/6/2021',periods=1000),
                  columns=list('ABCD'))
df1.cumsum().plot()


# 2、条形图
df2 = pd.DataFrame(data = np.random.rand(10,4),
                   columns = list('ABCD'))
df2


df2.plot.bar(stacked = True) # stacked 是否堆叠


# 3、饼图，百分比，自动计算
df3 = pd.DataFrame(data = np.random.rand(4,2),
                   index = list('ABCD'),
                   columns=['One','Two'])
# subplots两个图，多个图
# figsize 尺寸
df3.plot.pie(subplots = True,figsize = (8,8),colors = np.random.random(size = (4,3)))

# 4、散点图，横纵坐标，表示：两个属性之间的关系
df4 = pd.DataFrame(np.random.randint(0,50, size = (50,4)), columns=list('ABCD'))
df4.plot.scatter(x='A', y='B') # A和B关系绘制
# # 在一张图中绘制AC散点图，同时绘制BD散点图
ax = df4.plot.scatter(x='A', y='C', color='DarkBlue', label='Group 1');
df4.plot.scatter(x='B', y='D', color='DarkGreen', label='Group 2', ax=ax)
# # 气泡图，散点有大小之分
df4.plot.scatter(x='A',y='B',s = df4['C']*200)


df4['F'] = df4['C'].map(lambda x : x + np.random.randint(-5,5,size =1)[0])
df4
df4.plot.scatter(x = 'C',y = 'F')
plt.show()


# 5、面积图
df5 = pd.DataFrame(data = np.random.rand(10, 4), 
                   columns=list('ABCD'))
df5.plot.area(stacked = True,color = np.random.rand(4,3));# stacked 是否堆叠
plt.show()
# 颜色：红绿蓝
np.random.rand(4,3) # 4行3列

""" 
matplotlib 图形绘制


"""
# 基础知识
x = np.linspace(0,2*np.pi,100)
# 正弦函数  sin(x)
y = np.sin(x)
plt.plot(x,y)


# 调整横坐标和纵坐标
plt.xlim( -1, 10)
plt.ylim( -1.5, 1.5)
# 设置网格线
plt.grid(color="r",alpha = 0.5,linestyle = '--')

plt.show()


""" 
坐标轴的刻度 标签 标题
"""
plt.plot( x, y)

plt.title("sin(x)--zhaokai",fontsize = 20,color = 'r',pad=20)
plt.show()

# 如何设置中文字体
from matplotlib import font_manager
fm = font_manager.FontManager()
[font.name for font in fm.ttflist] # 查看所有字体
# 设置数字的负号
plt.rcParams['axes.unicode_minus'] = False
# 设置字的大小
plt.rcParams["font.size"] = 28
# 设置整个图片的大小
plt.figure(figsize=(12,9))

x = np.linspace(0,2*np.pi,100)
# 正弦函数  sin(x)
y = np.sin(x)
plt.plot( x, y)
plt.rcParams['font.family'] = 'KaiTi' # 设置中文字体
plt.title("赵凯",fontsize = 20,color = 'r',pad=20)
plt.show()


# 横坐标和纵坐标的标签
plt.xlabel("x")
plt.ylabel("f(x)=sin(x)",rotation = 0,horizontalalignment ="right" )

# 刻度
plt.yticks([-1,0,1])
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],color = 'r')


""" 
图例

"""
x =np.linspace(0,2*np.pi,100) # 横坐标
y = np.sin(x) # 纵坐标

# 绘制线型图
# 调整尺寸
plt.figure(figsize=(9,6))
plt.plot(x,y,label = 'sin(x)')
# 设置数字的负号
plt.rcParams['axes.unicode_minus'] = False


# 在一幅图中画多条线
plt.plot(x,np.cos(x),color = 'r',label = 'cos(x)')
plt.plot(x,np.tan(x),color = 'g',label = 'tan(x)')
plt.plot(x,np.sin(x)+np.cos(x),color = 'b',label = 'sin(x)+cos(x)')

# plt.show()

###  图例
plt.legend(["cos(x)","tan(x)","sin(x)+cos(x)"],
           fontsize=10,
           loc = 'upper left', # 位置
           ncol = 3, # 列数
           bbox_to_anchor=(0.1,1.1) # 图例的位置 x,y,width,height
           )
plt.show()


""" 
脊柱移动(坐标轴的移动)

"""
x = np.linspace(-np.pi,np.pi,50)
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(9,6),facecolor='green')

# plot 绘制凉饿图像 x-y成对出现 可以！！！
plt.plot(x,np.sin(x),np.cos(x))
# plt.show()

# 获取当前视图
ax = plt.gca()
# 调整整个图片的颜色
ax.set_facecolor("red")

ax.spines["right"].set_alpha(0)

# 右边和上面脊柱消失
ax.spines['right'].set_color('white') # 白色
ax.spines['top'].set_color('#FFFFFF') # rgb白色

# # 设置下面左边脊柱位置，data表示数据，axes表示相对位置0~1
ax.spines['bottom'].set_position(('data',0)) # 中间，竖直中间
ax.spines['left'].set_position(('data',0)) # 水平中间

# 加网格线
plt.grid()

plt.show()

# 查看有多少颜色
plt.colormaps()

# 图片的保存
plt.savefig(r".\1.png")  # dpi 调整像素大小 可以保存pdf等等
