#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:13:50 2022

@author: zhaokai
"""

#  字符串格式化打印
# % 占位符  1
name = "zhaokai"
age = 27
print("我叫%s,今年%i岁" % (name,age) )

print("我叫{},今年{}岁".format(name,age) )

print(f"我叫{name},今年{age}岁" )

print("%10d" % 90)

print("%.3f" % 3.1415926)

print("%10.3f" % 3.1415926)

print("{0:.3}".format(3.01415926))

print("{0:.3f}".format(3.01415926))

print("{0:10.3f}".format(3.01415926))


# 字符串的编码和解码
s = "天涯共此时"
print(s.encode(encoding="gbk"))

print(s.encode(encoding="utf-8"))

k = s.encode(encoding="gbk")
print(k.decode(encoding="gbk"))
  
# 函数
def calc(a,b):
    c = a+b
    return c
result = calc(100,200)
 

def fun(a,b,c):
    print("a=",a)
    print("b=",b)
    print("c=",c)

fun(10,20,30)
lst = [11,22,33]
fun(*lst)
# 字典 需要使用  **

def fun(a,b,c=10):
        print("a=",a)
        print("b=",b)
        print("c=",c)
fun(10,100)


def fun5(a,b,*,c,d,**args):
      pass
def fun6(*args,**args2):
    pass
def fun7(a,b=10,*aigs,**args2):
    pass


def fun(a,b):
    c = a+b # c就是局部变量
    
    print(c)

def fun1():
    global age
    age = 27
    print(age)
fun1()
print(age)  

def fac(n):
    if n ==1:
        return 1
    else:
        return n*fac(n-1)
print(fac(6))   

def fib(n):
    if n==1:
        return 1
    elif n==2:
        return 1
    else:
        return fib(n-1) + fib(n-2)
fib(6)    

for i in range(1,7):
    print(fib(i))


name = input("请输入你要查询的演员：")
for i in lst:
    actors = i["actors"]
    if name in actors:
        print(name+"出演了:"+i["title"])


# 异常处理
try :
    a = int(input("请输入第一个整数："))
    b = int(input("请输入第二个整数："))
    result  =  a/b
    print(result)
except ZeroDivisionError:
     print("对不起！输入的除数不能为0")
except ValueError:
    print("不能将字符串转换为数字")   
print("程序结束")
     
try :
    a = int(input("请输入第一个整数："))
    b = int(input("请输入第二个整数："))
    result  =  a/b
   
except BaseException as e:
    print("出错了",e)   
else:
    print("计算结果为：",result) 
    
try :
    a = int(input("请输入第一个整数："))
    b = int(input("请输入第二个整数："))
    result  =  a/b
   
except BaseException as e:
    print("出错了",e)   
else:
    print("计算结果为：",result) 
finally:
    print("无论出不出错，总会被执行的代码")
print("程序结束")


# traceback 模款的使用 
import traceback as tr
try:
    print("------------")
    print(10/0)
except:
    tr.print_exc()

    
# 代码调试
i = 0
while i <10:
    i +=1
    print(i)
 
class Student:
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def eat(self):
        print(self.name+"在吃饭")
stu1 = Student("掌扇", 20)
stu2 = Student("里斯", 21)        
 
print("------为stu2动态绑定性别属性---------")   
stu2.gender="女"
print(stu1.name,stu1.age)
print(stu2.name,stu2.age,stu2.gender)
 
print("=============")
def show():
    print("定义在类之外的，成为函数")
stu1.show = show
stu1.show()


class Student:
    def __init__(self,name,age):
        self.name = name
        self.__age = age
    def eat(self):
        print(self.name+"在吃饭")
    def show(self):
        print(self.name,self.__age)  
stu1 = Student("掌扇", 20)
stu2 = Student("里斯", 21)        
stu1.show()
print(stu1._Student__age)
print(dir(stu1))


class Person(object): #Person继承object类
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def info(self):
        print(self.name,self.age)

class Student(Person):
    def __init__(self,name,age,stu_no):
        super().__init__(name,age)
        self.stu_no=stu_no
    def info(self):
        super().info()
        print(self.stu_no)

class Teacher(Person):
    def __init__(self,name,age,teachofyear):
        super().__init__(name,age)
        self.teachofyear=teachofyear
    def info(self):
        super().info()
        print('教龄',self.teachofyear)

stu=Student('张三',20,'1001')
teacher=Teacher('李四',34,10)

stu.info()
print('----------------------')
teacher.info()



class Student:
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def __str__(self):
        return "我的名字是{0},今年{1}岁".format(self.name,self.age)

stu=Student('张三',20)
print(dir(stu))
print(type(stu))
print(stu)



a=20
b=100
c=a+b  #两个整数类型的对象的相加操作
d=a.__add__(b)

print(c)
print(d)

class Student:
    def __init__(self,name):
        self.name=name

    def __add__(self, other):
        return self.name+other.name

    def __len__(self):
        return len(self.name)

stu1=Student('Jack')
stu2=Student('李四')

s=stu1+stu2   #实现了两个对象的加法运算（因为在Student类中 编写__add__()特殊的方法）
print(s)
s=stu1.__add__(stu2)
print(s)
print('----------------------------------------')
lst=[11,22,33,44]
print(len(lst))  #len是内容函数len
print(lst.__len__())
print(len(stu1))
 
#北京地铁1号线运行图
print('地铁1\t\t\t四惠东→苹果园')
print('\t\t首车:05:05')
print('\t\t末车:23:30\t\t票价:起步价:2元')
print('------------------------------------------------------------------------------------------------')
print('  1\t\t   3\t  5\t   \t  7\t\t9\t\t  11\t\t 12\t\t 14\t\t  16\t\t18\t\t20')
print('  ⇌\t\t  ⇌\t\t  ⇌\t\t  ⇌\t\t⇌\t\t  ⇌\t\t\t ⇌\t\t ⇌\t\t  ⇌\t\t\t⇌\t\t⇌')
print('四惠东\t大望路\t永安里\t东单\t  天安门东\t 西单\t\t复兴门\t 木樨地\t公主坟\t  五棵松\t  八宝山')

# 输出杨老师出版书籍
book_name='Java程序设计教程'
publish='西安电子科技大学出版社'
pub_date='2019-02-02'
price=56.8
name = "zhaokai"
print('▶→→→→→→→→→→→→→→→→→→◀')
print('▷\t\t《',book_name,'》\t\t ◁')
print('▷\t出版社:',publish,'\t ◁')
print('▷\t出版时间:',pub_date,'\t\t\t ◁')
print('▷\t定  价:',price,'\t\t\t\t\t ◁')
print('\t\t测试：',publish,'\t\t')
print('\t\t测试：',name,'\t\t')
print("\t\tniubi ",name,"\t\t\t")
print('▶→→→→→→→→→→→→→→→→→→◀')

# 输出红楼梦金陵十三钗
'''1变量的赋值'''
name1='林黛玉'
name2='薛宝钗'
name3='贾元春'
name4='贾探春'
name5='史湘云'
print('❶\t'+name1)
print('❷\t'+name2)
print('❸\t'+name3)
print('❹\t'+name4)
print('❺\t'+name5)
print('-----------------------------------')
'''2种方式'''
lst_name=['林黛玉','薛宝钗','贾元春','贾探春','史湘云']
lst_sig=['❶','❷','❸','❹','❺']
for i in range(5):
    print(lst_sig[i],lst_name[i])
'''3字典'''
d={'❶':'林黛玉','❷':'薛宝钗','❸':'贾元春','❹':'贾探春','❺':'史湘云'}
print('''-------------------------------------''')
for key in d:
    print(key,d[key])
print('zip----------------')
for s,name in zip(lst_sig,lst_name):
    print(s,name)
# 显示各种颜色的格式
print("\033[0;35m\t\t赵凯牛逼\033[m")
print("\033[0;35m----------------------\033[m")
print("\033[0;32m\t\t赵凯牛逼\033[m")
print('\033[0;35m\t\t图书音像勋章\033[m')
print('\033[0;35m----------------------------------------\033[m')
print('\033[0;32m❀图书音像勋章\t\t✪专享活动\033[m')
print('\033[0;34m❤专属优惠\t\t\t☎优惠提醒\033[m')
print('\033[0;35m----------------------------------------\033[m')

h = 1.7
w = 70
bmi = w/(h*h)
print("您的身高是："+str(h))
print("您的体重是："+str(w))
print("您的BMI的指数是："+"{:0.2f}".format(bmi))




# 将指定的十进制的数转换为二进制、八进制、十六进制

def fun():
    num=int(input('请输入一个十进制的整数')) #将str类型转换成int类型
    print(num,'的二进制数为:',bin(num)) #第一种写法 ，使用了个数可变的位置参数
    print(str(num)+'的二进制数为:'+bin(num)) #第二种写法，使用"+"作为连接符  (+的左右均为str类型)
    print('%s的二进制数为:%s' % (num,bin(num))) #第三种写法，格式化字符串
    print('{0}的二进制数为:{1}'.format(num,bin(num))) #第三种写法，格式化字符串
    print(f'{num}的二进制数为:{bin(num)}') #第三种写法，格式化字符串
    print('-------------------------------------------------')
    print(f'{num}的八进制数为:{oct(num)}')
    print(f'{num}的十六进制数为:{hex(num)}')

if __name__ == '__main__':
    while True:
        try:
            fun()
            break
        except:
            print('只能输入整数!，程序出错，请重新输入')


# 为自己的手机充值
print("用户手机账户原有的话费为：\033[0;35m8元\033[m")
money = int(input("请输入用户充值金额：")) 
money +=8
print("当前的余额为：\033[0;35m",money,"元\033[m")


# 计算能量的消耗
num = int(input("请输入您当天行走的步数："))
calorie = num*28
print(f"今天共消耗了卡路里{calorie},即{calorie/1000}千卡")


# 预测未来子女的身高
f_h = float(input("请输入父亲的身高："))
m_h = float(input("请输入母亲的身高："))
c_h = (f_h + m_h)*0.54
print("预测子女的身高为：{}cm".format(c_h))


# 支付密码的验证
pwd = input("请输入你的支付宝密码：")
if pwd.isdigit():
    print("支付密码输入合法")
else:
    print("支付数字不合法：支付密码只能是数据")
print("------------------------")
print("支付数据合法" if pwd.isdigit() else "支付数字不合法")    

# 模拟qq登陆
qq = int(input("请输入你的qq号："))
pwd = input("请输入密码:")
if qq==1365980632 and pwd == "123":
    print("登陆成功")
else:
    print("对不起，您输入的账户密码不正确")

# 商品价格的大猜
import random
price = random.randint(1000, 1500)
print("今日猜的是小米草地机器人：价格在[1000-1500之间]")
guess = int(input("请输入："))
if guess > price:
    print("大了")
elif guess < price:
    print("小了")
else:
    print("恭喜你，猜对了")
print("该商品的真实价格为：",price) 
 

# 根据星座查询运势
d={
    '白羊座':'''本月贵人星座：水瓶座
本月小人星座：双鱼座
本月需要特别关注的日子：4日、5日、10日、16日、24日
本月DO：满足自己的情绪需求，改善长期压抑的心情。
本月DON’T：切勿在冲动之下做决定，三月仍是适合“以静制动”的阶段。''',
    '金牛座':'''本月贵人星座：金牛座
本月小人星座：白羊座
本月需要特别关注的日子：4日、5日、10日、22日、31日
本月DO：关注情绪需求，留意与未来有关的目标。
本月DON’T：信任自己的能力，切勿轻信他人的承诺。''',
    '双子座':'''本月贵人星座：白羊座
本月小人星座：水瓶座
本月需要特别关注的日子：4日、10日、16日、22日、24日
本月DO：要学会面对现实，接受长辈、业内前辈带来的帮助，切勿有过多无谓的“自尊心”。
本月DON'T：减少侥幸心理，这并非是拼好运的时期。''',
    '巨蟹座':'''本月贵人星座：巨蟹座
本月小人星座：魔羯座
本月需要特别关注的日子：4日、5日、10日、22日、24日
本月DO：培养自制力，规划自己的生活。
本月DON'T：最大的敌人是自己，要正视自己的惰性。''',
    '狮子座':'''本月贵人星座：射手座
本月小人星座：天蝎座
本月需要特别关注的日子：4日、5日、10日、22日、31日
本月DO：接受不完美的人生，才能改变的机会。
本月DON'T：关注与身边人的沟通方式，尽量别对往事翻旧账。''',
    '处女座':'''本月贵人星座：天秤座
本月小人星座：双子座
本月需要特别关注的日子：4日、10日、16日、24日、31日
本月DO：留意沟通方式，切勿口是心非。
本月DON’T：尽量别因为情绪方面的问题，迁怒于身边人。''',
    '天秤座':'''本月贵人星座：巨蟹座
本月小人星座：射手座
本月需要特别关注的日子：4日、5日、16日、22日、24日
本月DO：做任何事之前，先学会取悦自己。
本月DON'T：每个人都要有自己的因果，不必总是为身边人解决麻烦，必要时应当自私一些，也让他人学会为自己的人生负责。''',
    '天蝎座':'''本月贵人星座：双鱼座
本月小人星座：狮子座
本月需要特别关注的日子：4日、5日、10日、22日、24日
本月DO：关注情绪问题，寻找恰当的减压方式。
本月DON’T：重视个人健康，身体不适时务必要及时就医，切勿拖延。''',
    '射手座':'''本月贵人星座：双子座
本月小人星座：天秤座
本月需要特别关注的日子：5日、10日、16日、20日、24日
本月DO：关注与身边人的沟通方式，改善浮躁的情绪问题。
本月DON'T：近期会有较多的支出，可能会因此借贷或使用信用卡，要注意额度问题，切勿超出预算。''',
    '摩羯座':'''本月贵人星座：天蝎座
本月小人星座：巨蟹座
本月需要特别关注的日子：4日、5日、10日、22日、24日
本月DO：先照顾好自己，你才有帮助他人的能力及实力，否则都是空谈。
本月DON'T：改善不良生活方式，要学会重新为自己定义人生，切勿再得过且过。''',
    '水瓶座':'''本月贵人星座：狮子座
本月小人星座：处女座
本月需要特别关注的日子：4日、5日、10日、16日、22日
本月DO：接受环境带来的改变，调整自己的人生方向。
本月DON'T：正视身边人的需求，改善逃避问题的习惯。''',
    '双鱼座':'''本月贵人星座：处女座
本月小人星座：金牛座
本月需要特别关注的日子：4日、5日、10日、16日、24日
本月DO：调整情绪问题，用恰当的方式应对压力。
本月DON'T：正视环境带来的压力，必要时可以接受心理医生的帮助。'''
}
shar = input("请输入你的星座来查看近来的运势：")
print(d.get(shar))


# 循环输出26个字母对应的asc｜｜值
x = 97
for _ in range(1,27):
    print(chr(x),"--->",x)
    x +=1

x = 97 
while x<123:
   print(chr(x),"--->",x)
   x +=1



# 模拟用户登录
for i in range(1,4):
     user_name = input("请输入用户名：")
     user_pwd = input("请输入密码：")
     if user_name =="zhaokai" and user_pwd =="8888":
         print("恭喜你，登陆成功")
         break 
     else:
          print("输入的用户名或者密码错误")
          if i < 3:
             print(f"你还有{3-i}次机会")
else:
    print("对不起，三次输入错误，请联系后台管理员！！！")         


# 猜数字的游戏
import random
rand = random.randint(0,101)
for i  in range(1,11):
    num = int(input("在我心里面有一个数字（0-100），请你猜一猜"))
    if num >rand :
        print("你猜的数字大了")
    elif num<rand :
        print("你猜的数字小了")
    else:
        print("恭喜你，猜对了")
        break
print(f"你一共猜了{i}次")
if i <=3:
   print("你非常聪明")
elif i<=5:
   print("还可以嘛  小伙子")
elif i<=7:
   print("你小子，怎么这么笨")
else:
   print("猪🐷啊你 ")    

# 水仙花
import  math
for i in range(100,1000):
    if math.pow((i%10),3)+math.pow((i//10%10),3)+math.pow(i//100,3)==i:
        print(i)        


# 千年虫问题
year = [82,84,93,94,95,0,72,73,75]
print("原列表：",year)
for index,value in enumerate(year):
    if str(value) != "0":
        year[index] = int("19"+str(value))
    else:
        year[index] = int("200"+str(value))
print("修改之后的列表：",year)
year.sort()
print("排序之后的列表：",year)

# 商品的入库出库
lst = []
for i in range(0,5):
    goods = input("请输入商品的编号和商品的名称入库，每次只能输入一件商品：\n")
    lst.append(goods)
for m in lst:
    print(m)  
    
cart = []
while True:
    num = input("请输入要购买的商品编号：")
    for i in lst:
        if i.find(num)!=-1:
            cart.append(i)
            break
    if num =="q":
        break
print("您的购物车里面已经选好的东西为：")
# for m in cart:
#     print(m)
for m in  range(len(cart)-1,-1,-1):
    print(cart(m))      
            
        
        












          