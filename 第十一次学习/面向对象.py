# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:56:58 2022

@author: zhaokai

面向对象--C++、java、python  发号命令 ，输出结果

面向过程--C、Fartran       一步一步往下走




类 （class）:对具有相同属性和方法的一组对象的描述或者定义
    class 类的名称（）：
         ""类的说明"
         def_init_(self,...):
             pass
         ...
我的理解：模块是类的上级概念
 
"""
class Car():   #创建一个类，类的名称为Car，通常按规范，类的名称首字母大写
    """ 汽车目前价值估计程序"""
    def __init__(self,make,model,year):   #当类Car被创建一个对象时，Python会自动运行该方法。
                                        # 下划线是一种约定，避免Python默认的方法与普通的方法名称冲突
                                        #self必不可少，且必须位于其他形参的前面。
        self.make = make   #可通过对象访问的变量称为属性
        self.model = model
        self.year = year
        self.this_year  = 2018
    def mod_this_year(self,new_year):
        self.this_year = new_year

    def detection(self):  #方法detection
        duration = self.this_year - self.year
        price  =  30 - 2 * duration
        long_name = "你的" + self.make + self.model + "到目前已经行驶了" + str(duration)\
                    +"年,"+"目前价值" +str(price)+ "万"
        return long_name

class ElectricCar(Car):     #父类须位于子类前面
    def __init__(self,make,model,year):
            # super函数将父类和子类关联起来。让python调用ElectricCar的父类的方法__init__()
        super().__init__(make,model,year)  #父类称为超类，super因此而得名。
    def battery(self,capacity):
        self.capacity_num = capacity
        print("您选择的电池容量为:",self.capacity_num,"kWh")
    def detection(self):  #重写父类
        duration = self.this_year - self.year
        price = 30 - duration - (500/self.capacity_num )
        long_name = "你的" + self.make + self.model + "到目前已经行驶了" + str(duration) \
                    + "年," + "目前价值" + str(price) + "万"
        return long_name

# from car_file import *

my_car =  Car("Audi "," A4",2013)
my_car.mod_this_year(2020)
print(my_car.this_year)
result = my_car.detection()
print(result)
my_tesla = ElectricCar("Tesla ","model s",2017)
print(my_tesla.year)
my_tesla.battery(1000)
result = my_tesla.detection()
print(result)
# print(my_tesla.year,my_tesla.this_year)
# print(my_tesla.detection())






#
#
#
# """
# 库：一些经常使用的，经过检验的规范化程序或者子程序的集合
#    1.标准库：程序语言自身拥有的库  可以直接使用 无需安装
#    time库---获取时间
#    random库---随机数
#    turtle库---图形绘制库
#
#
#
#    2.第三方库： 第三方使用程序语言提供的程序库
#
# """
# import time
#
#
# lctime = time.localtime()
# print(lctime)
#
#
# import random
# print(random.randint(1, 10))
#
#
# # #  绘制太极图函数
# import turtle
# turtle.screensize(800, 600)  # 画布长、宽、背景色 长宽单位为像素
# turtle.pensize(1)  # 画笔宽度
# turtle.pencolor('black')  # 画笔颜色
# turtle.speed(10)  # 画笔移动速度
# TJT_color = {1: 'white', -1: 'black'}  # 太极图填充色 1 白色 -1 黑色
# color_list = [1, -1]
# R = 100  # 太极图半径
# for c in color_list:
#     turtle.fillcolor(TJT_color.get(c))  # 获取该半边的填充色
#     turtle.begin_fill()  # 开始填充
#     turtle.circle(R / 2, 180)
#     turtle.circle(R, 180)
#     turtle.circle(R / 2, -180)
#     turtle.end_fill()  # 结束填充 上色完成
#     turtle.penup()  # 提起画笔，移动不留痕
#     turtle.goto(0, R / 3 * c)  # 移动到该半边的鱼眼的圆上 R/3*c 表示移动到哪边
#     turtle.pendown()  # 放下画笔，移动留痕
#     turtle.fillcolor(TJT_color.get(-c))  # 获取鱼眼填充色, 与该半边相反
#     turtle.begin_fill()
#     turtle.circle(-R / 6, 360)
#     turtle.end_fill()
#     turtle.penup()
#     turtle.goto(0, 0)
#     turtle.pendown()
# turtle.penup()
# turtle.goto(0, -R - 50)
# turtle.pendown()
# turtle.done()
# input('Press Enter to exit...')  # 防止程序运行完成后就自动关闭窗口


"""
 time库---获取时间
"""
# import time
# time.time()：获取当前时间戳,代表着如今的时间与1970年1月1日0时0分0秒的时间差(以秒为单位)
# print(time.time())
# first_time = time.time()
# a = 0
# for i in range(10000000):
#     a += 1
# print(a)
# last_time = time.time()
# print(last_time - first_time)

# time.gmtime(secs)：获取当前时间戳对应的struct_time对象
# print(time.gmtime())

# time.localtime(secs)获取当前时间戳对应的本地时间的struct_time对象
# print(time.localtime())

# time.ctime(secs)获取当前时间戳对应的易读字符串表示，内部会调用time.localtime()函数以输出当地时间。
# print(time.ctime())

#time.mktime(t) 将struct_time对象t转换为时间戳，注意t代表当地时间。
# t = time.localtime()
# print(t)
# print(time.mktime(t))

#time.strftime()函数是时间格式化最有效的方法，几乎可以以任何通用格式输出时间。该方法利用一个格式字符串，对时间格式进行表达。
# t = time.localtime()
# a = time.strftime('%Y##%m-%d %H:%M:%S',t)
# print(a)

# strptime()方法与strftime()方法完全相反，用于提取字符串中时间来生成strut_time对象，可以很灵活的作为time模块的输入接口
# timeString = '2018-01-26 12:55:20'
# a = time.strptime(timeString, "%Y-%m-%d %H:%M:%S")
# print(a)

# sleep() 函数推迟调用线程的运行，可通过参数secs指秒数，表示进程挂起(睡眠)的时间。
# start_time = time.time()
# time.sleep(10)
# last_time = time.time()
# print(last_time - start_time)

# time.perf_counter()返回一个性能计数器的值(在分秒内)，即一个具有最高可用分辨率的时钟，以测量短时间。它包括了在睡眠期间的时间，并且是系统范围的。返回值的引用点是未定义的，因此只有连续调用的结果之间的差异是有效的
# a1 = time.perf_counter()
# # a2 = time.perf_counter()
# # print(a1 ,a2)
# b1 = time.time()
# a2 = time.perf_counter()
# b2 = time.time()
# print('perf_counter起始时间:',a1,'结束时间:',a2,'间距:',a2-a1)
# print('time起始时间:',b1,'结束时间:',b2,'间距:',b2-b1)

#模拟进度条显示
# scale = 50
# print('----------程序开始执行')
# start_time = time.perf_counter()
# for i in range(scale+1):
#     a = '*'*i
#     b = '.'*(scale-i)
#     c = (i/scale)*100
#     dur = time.perf_counter() - start_time
#     print('\r{:^3.0f}%[{}->{}]{:.2f}s'.format(c,a,b,dur),end='') #\r 表示将光标的位置回退到本行的开头位置
#     time.sleep(0.1)
# print('\n----------程序执行结束')







import random
# random()生成一个[0.0, 1.0)之间的随机小数
# a = random.random()
# print(a)

# seed()初始化随机数种子，默认值为当前系统时间
# random.seed()
# a = random.random()
# print(a)

# randint(a, b)生成一个[a,b]之间的整数
# a = random.randint(1,5)
# print(a)

# randrange(start, stop[, step])生成一个[start, stop)之间以step为步数的随机整数
# print(random.randrange(1,10))
# print(random.randrange(1,10,2))  #[1,3,5,7,9]
# print(random.randrange(1,10,3))  #[1,4,7]

# uniform(a, b)生成一个[a, b]之间的随机小数
# print(random.uniform(1,5))

# choice(seq)从序列类型(例如：列表)中随机返回一个元素
# a = ['剪子','石头','布']
# print(random.choice(a))

# 剪子石头布游戏
# ls = ['剪子','石头','布']
# a = random.choice(ls)
# b = input("请在'剪子','石头','布'中选择您的手势:")
# print('机器选择了：',a)
# if a == '剪子' and b == '剪子':
#     print('平局！')
# if a == '剪子' and b == '石头':
#     print('你输啦！')
# if a == '剪子' and b == '布':
#     print('你赢啦！')
# if a == '石头' and b == '剪子':
#     print('你赢啦！')
# if a == '石头' and b == '石头':
#     print('平局！')
# if a == '石头' and b == '布':
#     print('你输啦！')
# if a == '布' and b == '剪子':
#     print('你输啦！')
# if a == '布' and b == '石头':
#     print('你赢啦！')
# if a == '布' and b == '布':
#     print('平局！')

# shuffle(seq)将序列类型中元素随机排列，返回打乱后的序列
# a = ['剪子','石头','布']
# random.shuffle(a)
# print(a)

# sample(pop, k)从pop类型中随机选取k个元素，以列表类型返回
# a = ['剪子','石头','布']
# print(random.sample(a,2))




import turtle
# 设置主窗体的大小和位置
turtle.setup()
# turtle.done()

# forward() 沿着当前方向前进指定距离
# turtle.forward(200)
# turtle.fd(200)

# backward() 沿着当前相反方向后退指定距离
# turtle.backward(200)

# right(angle) 向右旋转angle角度
# turtle.right(90)
# turtle.fd(100)

# left(angle) 向左旋转angle角度
# turtle.left(45)
# turtle.fd(100)

# setheading(angle) 设置当前朝向为angle角度
# turtle.setheading(120)
# turtle.fd(100)
# turtle.left(45)
# turtle.fd(100)
# turtle.left(45)
# turtle.fd(100)
# turtle.setheading(120)
# turtle.fd(100)

# goto(x,y) 移动到绝对坐标（x,y）处
# turtle.goto(100,100)
# turtle.goto(100,-100)

# circle(radius,e) 绘制一个指定半径r和角度e的圆或弧形
# turtle.circle(120,360)

# undo() 撤销画笔最后一步动作
# turtle.undo()

# speed() 设置画笔的绘制速度，参数为0-10之间
# turtle.speed(10)
# turtle.fd(200)

#绘制六边形
# import turtle
# for i in range(3):
#     turtle.seth(i*120)
#     turtle.fd(100 )


# penup()提起画笔
# turtle.penup()
# turtle.forward(20)

# pendown() 放下画笔，与penup()配对使用
# turtle.pendown()
# turtle.circle(50,360)

# pensize(width)设置画笔线条的粗细为指定大小
# turtle.pensize(10)
# turtle.fd(100)
# turtle.pensize(1)
# turtle.fd(100)

# color()设置画笔的颜色
# turtle.color('red')
# turtle.fd(100)

# begin_fill()填充图形前，调用该方法
# turtle.begin_fill()
# turtle.color('red')
# turtle.circle(50,360)
# turtle.end_fill()

# filling()返回填充的状态，True为填充，False为未填充
# print(turtle.filling())

# clear()清空当前窗口，但不改变当前画笔的位置
# turtle.fd(200)
# turtle.clear()

# reset清空当前窗口，并重置位置等状态为默认值
# turtle.reset()

# screensize()设置画布的长和宽
# turtle.fd(1000)
# turtle.screensize(3000,3000)

# hideturtle()隐藏画笔的turtle形状
# turtle.fd(100)
# turtle.hideturtle()

# showturtle()显示画笔的turtle形状
# time.sleep(3)
# turtle.showturtle()

# isvisible()如果turtle可见，则返回True
# print(turtle.isvisible())
#
# turtle.done()





#绘制丘比特之心
# import turtle
# def Peach_heart():
#     turtle.left(135)
#     turtle.fd(100)
#     turtle.right(180)
#     turtle.circle(50,-180)
#     turtle.left(90)
#     turtle.circle(50,-180)
#     turtle.right(180)
#     turtle.fd(100)
# Peach_heart()
# turtle.penup()
# turtle.goto(100,30)
# turtle.pendown()
# turtle.seth(0)
# Peach_heart()
# turtle.penup()
# turtle.goto(-100,30)
# turtle.pendown()
# turtle.seth(25)
# turtle.fd(350)
# turtle.done()

# import turtle as tt
# from random import randint
# tt.TurtleScreen._RUNNING = True
# tt.speed(0)  # 绘图速度为最快
# tt.bgcolor("black")  # 背景色为黑色
# tt.setpos(-25, 25)  # 改变初始位置，这可以让图案居中
# tt.colormode(255)  # 颜色模式为真彩色
# cnt = 0
# while cnt < 500:
#     r = randint(0, 255)
#     g = randint(0, 255)
#     b = randint(0, 255)
#     tt.pencolor(r, g, b)  # 画笔颜色每次随机
#     tt.forward(50 + cnt)
#     tt.right(91)
#     cnt += 1
# tt.done()





