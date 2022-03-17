# -*- coding:utf-8 -*-
# os  与系统进行交互的包
# import os
# os.

# from os import system
# system("df -h")

# from os import *   导入所有方法

# from os.xx.xx import xx as rename   导入之后重新命名

# module.xxx   调用模块

# import my_first_modle
# my_first_modle.sayhi()

# 添加一个目录  以便可以调用模块不出错
# import sys
# d6path = "xx/xxx/xx/xx/xx/"
# sys.path.append("d6path")

# 动态替换目录  超级有用
import os,sys
d6path = "xx/xxx/xx/xx/xx/"
print(__file__)
print(os.path.dirname(__file__))
base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append("base_path")

# 包的导入  init文件
# import sys
# print(sys.path)
# from xx（高一级目录）.次级目录.文件目录 import 文件名

