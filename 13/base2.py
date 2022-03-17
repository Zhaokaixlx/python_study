# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:56:24 2022

@author: 赵凯
"""
class CPU:
    pass
class Disk:
    pass
class Computer:
    def __init__(self,cpu,disk):
        self.cpu=cpu
        self.disk=disk

#(1)变量的赋值
cpu1=CPU()
cpu2=cpu1
print(cpu1,id(cpu1))
print(cpu2,id(cpu2))
#(2)类有浅拷贝
print('------------------------------')
disk=Disk()  #创建一个硬盘类的对象
computer=Computer(cpu1,disk)  #创建一个计算机类的对象

#浅拷贝
import  copy
print(disk)
computer2=copy.copy(computer)
print(computer,computer.cpu,computer.disk)
print(computer2,computer2.cpu,computer2.disk)
print('----------------------------------------')
#深拷贝
computer3=copy.deepcopy(computer)
print(computer,computer.cpu,computer.disk)
print(computer3,computer3.cpu,computer3.disk)

# 模块
# 导入  import  xxx
# from 模块名称 import 函数、变量、类
import math
print(id(math))
print(type(math))
print(math)
print(math.pi)

print(dir(math))

print(math.pow(2, 10))

from math import pi

print(pi*9)

# sys 模块
import sys,time
import urllib
print(sys.getsizeof(24))
print(sys.getsizeof(True))

print(time.time())
print(time.localtime(time.time()))

a = urllib.request.urlopen("http://www.baidu.com").read()
print(a)

import schedule,time

def job():
    print("赵凯，你怎么这么牛逼")
schedule.every(3).seconds.do(job)
while True:
    schedule.run_pending()
    time.sleep(1)

file = open(r"D:\pycharmcode\13\a.txt","r",encoding="utf-8")
print(file.readlines())
file.close()

file = open(r"D:\pycharmcode\13\a.txt","r",encoding="utf-8")

print(file.read(30))

# 一行
print(file.readline())
file.close()
# write(str)
# writelines()

# 移动光标 中文字符占2个字节
file = open(r"D:\pycharmcode\13\a.txt","r",encoding="utf-8")
file.seek(10)
print(file.read())
file.close()

# 文件指针的地方
print(file.tell())

# file.flush()
# 可以将缓冲区的内容写入文件
file = open(r"D:\pycharmcode\13\a.txt","a",encoding="utf-8")
file.write("zhaokai")
file.flush()
file.write(" niu bi a!")
file.close()

# with 语句 （上下文管理器）
with open(r"D:\pycharmcode\13\a.txt","r",encoding="utf-8") as file:
    print(file.read())

class Mycontentmgr:
     def __enter__(self):
         print("enter方法被执行了")
         return self
     def __exit__(self):
         print("exit方法被执行了")
     def show(self):
         print("show 方法被调用了")
with Mycontentmgr() as file:
    file.show()

with open(r"D:\pycharmcode\13\1.jpg","rb") as src_file:
    with open(r"D:\pycharmcode\13\1copy.jpg","wb") as t_file:
        t_file.write(src_file.read())

# os 模块
"""
os 模块是与操作系统相关的一个模块
"""
import os
os.system("notepad.exe")
os.system("calc.exe")
# 可以直接调用可执行文件
os.startfile(r"C:\Program Files (x86)\Tencent\WeChat\WeChat.exe")

# 打印当前的的工作路径
print(os.getcwd())
# 列出这里路径下的所有文件
lst = os.listdir("D://pycharmcode//13")
# 创建目录
os.mkdir("")
# 创建多级目录
os.makedirs("a/b/c")
# 删除目录
os.rmdir()
# 删除多级目录
os.removedirs()

# 将path 设置为当前的工作目录
os.chdir("path")


## os.path 模块操作目录相关的额函数
import os.path
# 绝对路径
print(os.path.abspath(r"C:\Program Files (x86)\Tencent\WeChat\WeChat.exe"))
# 判断文件或者目录是否存在
os.path.exists("path")
# 拼接
# os.path.join(path, paths)

# 文件的路径和文件的名字拆分
os.path.split()
os.path.splitext()

# 目录中提取文件名
os.path.basename("path")

os.path.dirname("path")

os.path.isdir("path")

"""
获取指定目录下的所有的python 文件
"""
import os
path = os.getcwd()
lst = os.listdir(path)
for filename in lst:
    if filename.endswith(".py"):
        print(filename)

import os
path = os.getcwd()
lst_files = os.walk(path)
print(lst_files)

for dirpath,dirname,filename in lst_files:
    print(dirpath)
    print(dirname)
    print(filename)
    print("++++++++++++")


















