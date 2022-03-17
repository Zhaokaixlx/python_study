# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 23:09:57 2022

@author: Administrator
"""

import sys
'''
sys库的作用：主要是针对与Python解释器相关的变量和方法，
            大白话：查看python解释器信息及传递信息给python解释器。
sys.argv    #获取命令行参数列表，第一个元素是程序本身
sys.exit(n) #退出Python程序，exit(0)表示正常退出。当参数非0时，会引发一个SystemExit异常，可以在程序中捕获该异常
sys.version #获取Python解释程器的版本信息
sys.maxsize #最大的Int值，64位平台是2**63 - 1
sys.path    #返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值
sys.platform    #返回操作系统平台名称
sys.stdin   #输入相关
sys.stdout  #输出相关
sys.stderr  #错误相关
sys.exc_info()  #返回异常信息三元元组
sys.getdefaultencoding()    #获取系统当前编码，默认为utf-8
sys.setdefaultencoding()    #设置系统的默认编码
sys.getfilesystemencoding() #获取文件系统使用编码方式，默认是utf-8
sys.modules #以字典的形式返回所有当前Python环境中已经导入的模块
sys.builtin_module_names    #返回一个列表，包含所有已经编译到Python解释器里的模块的名字
sys.copyright   #当前Python的版权信息
sys.flags   #命令行标识状态信息列表。只读。
sys.getrefcount(object) #返回对象的引用数量
sys.getrecursionlimit() #返回Python最大递归深度，默认1000
sys.getsizeof(object[, default])    #返回对象的大小
sys.getswitchinterval() #返回线程切换时间间隔，默认0.005秒
sys.setswitchinterval(interval) #设置线程切换的时间间隔，单位秒
sys.getwindowsversion() #返回当前windwos系统的版本信息
sys.hash_info   #返回Python默认的哈希方法的参数
sys.implementation  #当前正在运行的Python解释器的具体实现，比如CPython
sys.thread_info #当前线程信息
'''
'''
sys.argv说明
从程序外部获取参数的桥梁，获取命令行参数，返回一个列表，
其中包含了脚本路径及传递给 Python 脚本的命令行参数。
并非等用户输入，可以由系统传递给python脚本程序。
优点：方便程序员可以通过命令方式直接控制程序的运行状态，
      不需要使用input对数据进行处理
'''
print('开始执行程序')
print(sys.argv)       #argv[0]:为脚本的名称,argv[0]之后的内容为命令行参数
#
for i in range(int(sys.argv[1])):
    print('执行{}'.format(i))

def run1(num):
    print('开始启动发动机'+num)
    pass    #省略
run1(sys.argv[2])

'''
sys.exit()说明
程序退出，如果是正常退出是sys.exit(0)，
参数是可以任意填写，
但一般规定0为正常退出，非0为异常退出。
'''
print('hello')
def fun1():
    pass
    return 0  #如果函数检查到内存异常，需要退出该程序，return 1

sys.exit(0)
print('python')


print(sys.version)   #获取Python解释程序的版本信息
if sys.version < 2.7: 
    print("不能执行,请更新一下你的python解释器版本")










