# class Dog():
#     """一次模拟小狗的简单尝试"""
#     def __init__(self,name,age):
#         self.name = name
#         self.age = age
#     def sit(self):
#         """模拟小狗被命令时蹲下"""
#         print(self.name.title()+" is now sitting" )
#     def roll_over(self):
#         """模拟下沟被命令时打滚"""
#         print(self.name.title()+"rolled over!")
# my_dog = Dog("tai zi",4)
# print("我的狗狗的名字是"+my_dog.name.title()+".")
# print(my_dog.name)
# print(my_dog.age)
# print(my_dog.sit())
# print(my_dog.roll_over())

"""
描述汽车的类
"""
# class Car():
#     def __init__(self,make,model,year):
#         """初始化汽车属性"""
#         self.make = make
#         self.model = model
#         self.year = year
#         # 默认属性
#         self.odometer = 0
#     def get_descriptive_name(self):
#         """返回汽车的完整信息"""
#         long_name = str(self.year)+" "+self.make+" "+self.model
#         return long_name
#     def read_odometer(self):
#         """打印一下汽车的里程信息"""
#         print("This car has"+ str(self.odometer) +"mile")
# my_car = Car("audi","a6",2016)
# print(my_car.make)
# print(my_car.model)
# print(my_car.get_descriptive_name())

# """
# pip命令
# 在应用市场 搜索  下载  安装
# ｛pip命令就是下载库、包、模块｝
# """

"""
模块
包
库
"""
# import time
# print(time.time())
# dir(time)

# import random
# num = random.randrange(1,51)
# print(num)
# dir(random)

"""
re (正则表达式模块)
os(路径)
"""

# debug 断点：可以让我们直接跳到想要调试代码的地方
# def add(a,b):
#     x=a
#     y=b
#     z=x+y
#     print(z)
# add(1,2)
# add(2,3)
# add(3,4)

# 匿名函数: lambda f(x,y) = x+y
# sum2 = lambda x,y:x+y
# sum3 = lambda x,y:x*y
# print(sum2(1,2))
# print(sum3(3,4))
#
# sum4 = lambda x,y,z:x+y+z
# sum5 = lambda x,y,z:x*y*z
# print(sum4(1,2,3))
# print(sum5(3,4,5))

# 添加条件表达式

# 1.正常表达的
# def func1(x):
#     if x==1:
#         print("ok")
#     else:
#         print("no")
# func1(1)
# func1(2)
# # 2.使用匿名函数表达
# fun2 = lambda x:"ok" if x==1 else "no"
# func1(1)
# func1(2)

# map 函数  map(function,iterable,......)
# def square(x):
#     return x**2
# data = list(range(1,11))
# print(list(map(square,data)))
# print(list(map(lambda x:x**2,data)))
# print(list(map(lambda x,y,z:x*y*z,data,data,data)))




