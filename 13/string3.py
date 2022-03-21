# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:48:25 2022

@author: zhaokai
"""
# 创建一个乐器的类
class Instrument():
     def make_sound(self):
         pass
class Erhu(Instrument):
    def make_sound(self):
        print("二胡在发声!")
class Pinao(Instrument):
    def make_sound(self):
        print("钢琴在演奏！！")
class Violin(Instrument):
    def make_sound(self):
        print("小提琴在演奏！！")
class Bird():
    def make_sound(self):
        print("小鸟在唱歌！！")


# 演奏的函数
def play(Instrument):
    Instrument.make_sound()

if __name__ =="__main__":
    play(Erhu())
    play(Pinao())
    play(Violin())
    play(Bird())

# 用面向对象的思想，设计自定义的类
# 描述出租车和家用轿车的信息
class Car(object):
    def __init__(self,type,no):
        self.type = type
        self.no =no
    def Start(self):
        pass
    def Stop(self):
        pass
class Taxi(Car):
    def __init__(self,type,no,company):
        super().__init__(type,no)
        self.company = company
    def Start(self):
        print("乘客您好！！")
        print(f"我是{self.company}出租车公司的，我的车牌号时{self.no}，请问您是要去哪里？")
    def Stop(self):
        print("目的地到了，请您付款下车，欢迎下次乘坐！！")

class Familycar(Car):
    def __init__(self,type,no,name):
        super().__init__(type,no)
        self.name = name
    def Stop(self):
        print("目的地到了，我们去玩吧！！")
    def Start(self):
        print(f"我是{self.name}，我的汽车我做主")
if __name__ == "__main__":
    taxi = Taxi("上海大众", "京A8792744", "长城")
    taxi.Start()
    taxi.Stop()
    print("-"*30)
    Familycar = Familycar("广汽丰田","京8888888","武大郎")
    Familycar.Start()
    Familycar.Stop()

## 模拟高铁售票系统
import  prettytable as pt

def show_ticket(row_num):
    tb=pt.PrettyTable()
    tb.field_names=['行号','座位1','座位2','座位3','座位4','座位5']
    for i in range(row_num):
        lst=[f'第{i+1}行','有票','有票','有票','有票','有票']
        tb.add_row(lst)
    print(tb)

#订票
def order_ticket(row_num,row,column):
    tb=pt.PrettyTable()
    tb.field_names = ['行号', '座位1', '座位2', '座位3', '座位4', '座位5']
    for i in range(row_num):
        if int(row)==i+1:
            lst = [f'第{i + 1}行', '有票', '有票', '有票', '有票', '有票']
            lst[int(column)]='已售'
            tb.add_row(lst)
        else:
            lst = [f'第{i + 1}行', '有票', '有票', '有票', '有票', '有票']
            tb.add_row(lst)
    print(tb)

if __name__ == '__main__':
    row_num=13
    show_ticket(row_num)
    choose_num=input('请输入选择的座位,如13,5表示13排5号座位')
    try:
      row,column=choose_num.split(',')
    except:
        print('输入格式有误，如13排5号座位，应该输入13,5')
    order_ticket(row_num,row,column)

# # 推算几天之后的日期
import datetime

def inputdate():
    indate=input('请输入开始日期:(20200808)后按回车:')
    indate=indate.strip()
    datestr=indate[0:4]+'-'+indate[4:6]+'-'+indate[6:]
    return datetime.datetime.strptime(datestr,'%Y-%m-%d')

if __name__ == '__main__':
    print('-----------------推算几天后的日期----------------------------')
    sdate=inputdate()
    in_num=int(input('请输入间隔天数:'))
    fdate=sdate+datetime.timedelta(days=in_num)
    print('您推算的日期是:'+str(fdate).split(' ')[0])

# 记录用户登录日志
import time
def show_info():
    print('输入提示数字，执行相应操作: 0.退出  1.查看登录日志')
#读录日志
def write_loginfo(username):
    with open ('log.txt','a') as file:
        s='用户名名:{0},登录时间:{1}'.format(username,time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        file.write(s)
        file.write('\n')

#读取日志
def read_loginfo():
    with open('log.txt','r')as file:
        while True:
            line=file.readline()
            if line=='':
                break #退出循环s
            else:
                print(line,end='')#输出一行内容

#以主程序方式运行
if __name__ == '__main__':
    username=input('请输入用户名:')
    pwd=input('请输入密码:')
    if 'zhaokai'==username and 'zhaokai'==pwd:
        print('登录成功')
        write_loginfo(username) #写入日志
        show_info() #提示信息
        num=int(input('输入操作数字:'))
        while True:
            if num==0:
                print('退出成功')
                break
            elif num==1:
                print('查看登录日志')
                read_loginfo()
                show_info()
                num = int(input('输入操作数字:'))
            else:
                print('您输入的数字有误')
                show_info()
                num = int(input('输入操作数字:'))
    else:
        print('用户名或密码不正确 ')



# # 模拟淘宝客服自动回复
def find_answer(question):
    with open(r"D:\python_course\pycode\chap17\实操案例十五\replay.txt",encoding="gbk") as file:
        while True:
              line = file.readline()
              if not line: # if line ==""
                  break
              # 字符串的分割
              keyword = line.split("|")[0]
              reply = line.split("|")[1]
              if keyword in question:
                  return reply
    return False

if __name__  == "__main__":
    question=input('Hi,您好，小蜜在此等主人很久了，有什么烦恼快和小蜜说吧:  ')
    while True:
         if question=='bye':
             break
         reply= find_answer(question)
         if not reply:
             question=input('小蜜不知道你在说什么，您可以问一些关于订单、物流、账户、支付等问题,（退出请输入bye）')
         else:
             print(reply)
             question = input('小主，你还可以继续问一些关于订单、物流、账户、支付等问题（退出请输bye）')

    print("小主人 再见")







