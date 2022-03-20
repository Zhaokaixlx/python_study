#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 09:37:18 2022

@author: zhaokai
"""
##根据星座查询性格
#创建星座的列表
constellation=['白羊座','金牛座','双子座','巨蟹座','狮子座','处女座','天秤座','天蝎座','射手座','摩羯座','水瓶座','双鱼座']
#创建性格列表
nature=['积极乐观','固执内向','圆滑世故','多愁善感','迷之自信','精明计较','犹豫不决','阴暗消极','放荡不羁','务实本分','作天作地','安于现状']

# 把两个列表转化为字典
a = dict(zip(constellation,nature))

key= input("请输入你要查询的星座:")

print(key," 的性格特点为：",a.get(key))

flag=True
for item in a:
    if key==item:
        flag=True
        print(key,'的性格特点为:',a.get(key))
        break
    else:
        #print('对不起，您输入的星座有误')
        flag=False

if not flag:
    print('对不起，您输入的星座有误')


## 模拟12306 火车票下单
dict_ticket={'G1569':['北京南-天津南','18:05','18:39','00:34'],
             'G1567':['北京南-天津南','18:15','18:49','00:34'],
             'G8917':['北京南-天津西','18:20','19:19','00:59'],
             'G203 ':['北京南-天津南','18:35','19:09','00:34']}
print('车次\t\t出发站-到达站\t\t出发时间\t\t\t到达时间\t\t\t历时时长')    
for i in dict_ticket:
     print(i,end="")
     for m in dict_ticket[i]:
          print(m,end="\t\t")
     print()# 换行
# 输入购买的车次
train_no  =  input("请输入要购买的车次：")
person = input("请输入乘车人，如果是多人，请用逗号分割: ")
s = f"您已经购买了{train_no}次列车"
s_info =  dict_ticket[train_no] # 获取车次的详细信息
s += s_info[0] + " " + s_info[1] + "开"

print(f"{s}请{person}尽快取走纸质车票【铁路客服💁】")

# 改善

dict_ticket={'G1569':['北京南-天津南','18:05','18:39','00:34'],
             'G1567':['北京南-天津南','18:15','18:49','00:34'],
             'G8917':['北京南-天津西','18:20','19:19','00:59'],
             'G203 ':['北京南-天津南','18:35','19:09','00:34']}
print('车次\t\t出发站-到达站\t\t出发时间\t\t\t到达时间\t\t\t历时时长')    
for i in dict_ticket:
     print(i,end="")
     for m in dict_ticket[i]:
          print(m,end="\t\t")
     print()# 换行
# 输入购买的车次
train_no  =  input("请输入要购买的车次：")
person = input("请输入乘车人，如果是多人，请用逗号分割: ")

if train_no !="" and person !="":      
    s  = f"您已经购买了{train_no}次列车"
    s_info =  dict_ticket[train_no] # 获取车次的详细信息
    s += s_info[0] + " " + s_info[1] + "开"
    
    print(f"{s}请{person}尽快取走纸质车票【铁路客服💁】")
else:
    print("对不起你输入有误  请重新购买")




# 我的咖啡馆你做主
coff_name = ('蓝山','卡布奇诺','拿铁','皇家咖啡','女五咖啡','美丽与哀愁')
print("您好！ 欢迎光临小猫咖啡屋")
print("本店经营的咖啡有: ")
for index,item in enumerate(coff_name):
         print(index+1,"-",item,end=" ") 
index = int(input("\n请输入您喜欢的咖啡编号："))
if 0<=index<=len(coff_name):
   print(f"您的咖啡[{coff_name[index-1]}]好了，请您享用")
else:
   print("输入的编号不合法，请您重新输入！！！")


# 显示2019年中超联赛前5名排行
scores=(('广州恒大',72),('北京国安',70),('上海上港',66),('江苏苏宁',53),('山东鲁能',51))       
for index,item in enumerate(scores):
    print(index+1,".",end=" ")
    for score in item:
        print(score,end=" ")
    print() # 空行
    
# 模拟手机通讯录
phones  = set()
for i in range(1,6):
    info = input(f"请输入第{i}个朋友的姓名和手机号码： ")
    phones.add(info)
for m in phones:
    print(m)
    

# 统计字符串中出现指定字符的次数
def get_coun(s,ch):
    count = 0
    for i in s:
        if ch.upper() ==i or ch.lower():
            count +=1
    return count
if __name__=="__main__":
      s="helool,python,java,zhaokai,zhaoli,zhangxiue,zhaoyouyu,laolao" 
      ch = input("请输入您要统计的字符：")
      count = get_coun(s, ch)
      print(f"{ch}在{s}中出现的次数为：{count}")
     
        
# 格式化输出商品的名称和单价

def show(lst):
    for item in lst:
        for i in item:
            print(i,end="\t\t")
        print()
lst=[['01','电风扇','美的',500],
     ['02','洗衣机','TCL',1000],
     ['03','微波炉','老板',400] ]
print("编号\t\t名称\t\t\t品牌\t\t单价")
# for item in lst:
#     for i in item:
#         print(i,end="\t\t")
#     print()
show(lst)
print("------字符串的格式化-------------")
for item in lst:
    item[0] = "0000" + item[0]
    item[3] = "${:2f}".format(item[3])
# for item in lst:
#     for i in item:
#         print(i,end="\t\t")
#     print()
show(lst)
  

# 迷你计算器
def calc(a,b,op):
    if op =="+":
        add(a,b)
    elif op =="-":
        sub(a,b)
    elif op=="*":
        mul(a, b)
    elif op=="/":
        if b!=0:
            return div(a,b)
        else:
            return"除数不能为0"
            
def add(a,b):
    return a+b
def sub(a,b):
    return a-b
def mul(a,b):
    return a*b
def div(a,b):
    return a/b
if __name__ == "__main__":
    a  = int(input("请输入第一个整数："))
    b  = int(input("请输入第二个整数："))
    op = input("请输入运算符：")
    print(calc(a, b, op))
   
# 猜数字游戏
import random
def guess(num,guess_num):
    if num == guess_num :
        return 0
    elif guess_num>num:
        return 1
    else:
        return -1
num = random.randint(1,101)
for i in range(10):
    guess_num = int(input("我心里有个数字【0-100】的整数，请你猜一猜："))
    result = guess(num, guess_num)
    if result ==0:
        print("恭喜你，猜对了")
        break
    elif result >0:
        print("猜大了")
    else:
        print("猜小了")
else:
     print("你他妈的真是个猪🐷") 
     
     
# 编写程序输入学员成绩
try:
    
    score = int(input("请输入分数："))
    if 0<=score<=100:
        print("分数为：",score)
    else:
        raise Exception("丢你老母！！！")
except Exception as e:
    print(e) 
    
    
# 编写程序 判断三个参数能否构成三角形
def is_triangel(a,b,c):
    if a<0 or b<0 or c<0:
        raise Exception("三角形三条边不能是负数")
    # 判断是否构成三角形
    if a+b>c and b+c>a and a+c>b:
        print(f"三角形的边长为a={a},b={b},c={c}")
    else:
        raise Exception(f"a={a},b={b},c={c},不能构成三角形")
if __name__ =="__main__":
     try:
         a = int(input("请输入第一条边："))
         b = int(input("请输入第二条边："))
         c= int(input("请输入第三条边："))
         is_triangel(a, b, c)
     except Exception as e:
         print(e)
         
         
# 定义一个圆的类 计算面积和周长
import  math
class Circle(object):
    def __init__(self,r):
        self.r=r

    def get_area(self):
        return math.pi*math.pow(self.r,2)

    def get_perimeter(self):
        return 2*math.pi*self.r


if __name__ == '__main__':
    r=int(input('请输入圆的半径:'))
    c=Circle(r)
    print(f'圆的面积为:{c.get_area()}')
    print(f'圆的周长为:{c.get_perimeter()}')

    print('圆的面积为:{:.2f}'.format(c.get_area()))
    print('圆的周长为:{:.2f}'.format(c.get_perimeter()))         
    
# 定义学生类 录入5个学生信息储存在列表中
class Student(object):
    def __init__(self,stu_name,stu_age,stu_gender,stu_score):
        self.stu_name = stu_name
        self.stu_age = stu_age
        self.stu_gender = stu_gender
        self.stu_score = stu_score
    def show(self):
        print(self.stu_name,self.stu_age,self.stu_gender,self.stu_score)
if __name__ == "__main__":
   print("请输入五位学员的信息：（姓名#年龄#性别#成绩）")
   lst = []
   for i in range(1,6):
       s = input(f"请输入第{i}位学员的信息和成绩")
       s_lst = s.split("#")
       # 创建学生对象
       stu = Student(s_lst[0],int(s_lst[1]),s_lst[2],float(s_lst[3]))
       lst.append(stu)
   for item in lst:
        item.show()
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

