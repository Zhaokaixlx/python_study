# 赵凯
# 开发时间：2022/3/12-16:36
# --UTF-8--

# 可以输出数字
print(520)
print(98.5)
# 输出字符串
print("zhao kai ni hao")
# 输出含有运算符的表达式
print(3+5)

# 将数据输出的文件中
fp = open(r"D:\pycharmcode\13\text.txt","a+")
print("ni hao zhao kai ",file=fp)
fp.close()
# 不进行换行输出（输出内容在一行中）
print("ni","hao","python")

# 转义字符
print("hell0\nzhao kai")
print("hell0\tzhao kai")# 制表符 占4位
print("hell0\rzhao kai")# 回车
print("hell0\bzhao kai")# 返回一格

print("http:\\\\www.baidu.com")
print("老师说：\"zhao kai niu bi\"")
# 最后一个字符不能是一个反斜杠

# 查看系统内的保留字
import keyword
print(keyword.kwlist)

# 变量
name ="zhao kai"
print(name)
print("标识",id(name))
print("类型",type(name))
print("值",name)

# 整数类型
n1 = 98
print(n1,type(n1))
# 整数可以表示为二进制、十进制、八进制、十六进制
print("十进制",98)
print("二进制",0b100010)
print("八进制",0o176)
print("十六进制",0xef23)

# 浮点数
a = 3.1415926
print(a,type(a))
n1 = 1.1
n2 = 2.2
n3 = 2.1
print(n1 + n2)

from decimal import Decimal
print(Decimal("1.1")+Decimal("2.2"))

from decimal import Decimal
print(Decimal(n1)+Decimal(n2))

print(n1+n3)

# 布尔类型
f1 = True
f2 = False
print(f1,type(f1))
print(f2,type(f2))
# 布尔值可以转换为整数进行计算
print(f1+1)
print(f2+8)

# 类型转换
name = "zhao kai"
age = 27
print("我叫"+name+"，今年"+str(age)+"岁")



# input
present = input("name?")
print(present,type(present))

a = int(input("请输入一个加数："))
b = int(input("请输入另一个加数："))
print(a+b)

# 运算符
# 一正一负整除 向下取整
# 一正一负取余  除数-被除数-商


# 赋值运算符 右--左
a = 20
a +=20
print(a)
a = 20
a -=20
print(a)

a,b,c = 10,20,30
print(a+b+c)

a,b = 20,30
print("----交换之前-----",a,b)
a,b = b,a
print("----交换之后-----",a,b)

a,b = 20,30
print(a>b)
print(a<=b)
print(a!=b)

a,b = 10,10
print(a==b)
print(a is b)

print(id(a),id(b))

# is 判断 id值是否相同
list1 = [1,2,3,4]
list2 = [1,2,3,4]
print(list1 == list2)
print(list1 is list2)
print(id(list1),id(list2))

print(16 << 1) # 向左移动相当于*2
print(16 >> 1) # 向右移动相当于/2



# 条件语句  流程控制
print("---程序开始----")
print("1.打开冰箱门")
print("2.把大象放进去")
print("3.把冰箱门观念上")
print("----程序结束-----")

# 单分支
money = 10000
s = int(input("请输入取款金额："))
if s <= money:
    money = money - s
print("取款成功，余额为:",money)
# 双分支
num = int(input("请输入一个整数"))
if num%2 == 0:
    print(num,"是偶数")
else:
    print(num,"是奇数")
# 多分支
# 嵌套
a = input("你是否是会员？y/n")
m = int(input("原始金额是："))
if a == "y":
   if m >=200:
       print("你的付款金额为:",m*0.8)
   elif m>=100:
       print("你的付款金额为:",m*0.9)
   else:
       print("你的付款金额为:",m)
else:
    if m>=200:
        print("你的付款金额为:",m*0.95)
    else:
        print("你的付款金额为:",m)
# 条件表达式
num_a = int(input("请输入第一个整数"))
num_b = int(input("请输入第二个整数"))
# 比较大小
"""if num_a>=num_b:
    print("a>b")
else:
    print("a<b")
"""
print("a>b" if num_a >= num_b else "a<b")

# pass  ---什么都不做  占位符
a = input("你是否是会员？y/n")
m = int(input("原始金额是："))
if a == "y":
   pass
else:
    pass

# 循环结构
a = 1
if a<=10:
    print(a)
    a +=1
a = 1
while a <= 10:
    print(a)
    a += 1

a = 0
sum = 0
while a<5:
    sum +=a
    a +=1
print("和为",sum)

a = 1
sum = 0
while a<101:
    if a%2==0:
        sum +=a
    a +=1
print("和为",sum)

# for in 循环
for item in "python":
    print(item)

# 如果不使用自定义变量 "-"
for _ in range(5):
    print("人生苦短，woyongpython")
# 1-100的偶数和
sum = 0
for item in range(1,101):
    if item%2==0:
        sum +=item
print(sum)

for item in range(100,1000):
    ge = item%10
    shi = item//100%10
    bai = item//100
    # print(ge,shi,bai)
    if ge**3+shi**3+bai**3==item:
       print(item)

# 密码 break if语句
for i in range(3):
    s = input("请输入密码：")
    if s == "8888":
        print("输入正确")
        break
    else:
        print("输入错误")
    a +=1

# continue 语句
for i in range(1,51):
    if i%5 !=0:
        continue
    print(i)


# else 语句  for while 也可以用
# 当遇到break 不执行else  否则会执行
for i in range(3):
    s = input("请输入密码：")
    if s == "8888":
        print("输入正确")
        break
    else:
        print("输入错误")
else:
    print("对不起，三次密码都输错了！！！")

i =0
while i<3:
    s = input("请输入密码：")
    if s == "8888":
        print("输入正确")
        break
    else:
        print("输入错误")
        i +=1
else:
    print("对不起，三次密码都输错了！！！")


# 循环嵌套
for i in range(1,4):
    for j in range(1,5):
        print("+",end="\t")
    print()

# 直角三角形
for i in range(1,10):
    for j in range(1,i+1):
        print(i,"*",j,i*j,end="\t")
    print()

# 流程控制语句中的break 和 continue 在二层循环中的使用


# 列表
a = ["zhaokai","niu bi",99,88,22,33]
b = list(["zhaoli","mei mei",99])
print(a[0])

print(a.index("niu bi"))
print(a.index(22,1,7))

print(a[2])
print(a[-3])

print(a[0:3])
print(a[0:7:2])
print(a[::-1])

# in  not in
print(99 in a)
print(10 not in a)
# 遍历
for i in a:
    print(i)
# 增加
a.append(10)
a.append(b)
a.extend(b)
print(a)

a.insert(1, 9999)
print(a)

b[:1] = a
print(b)


# 删除
b.remove("mei mei")
b.pop(1)
# del
# clear

# 修改
b[1] = "ge ge"
print(b)
b[1:3] = [22,66,33,88]

# 排序
c = [0,1,100,2,5,99,8,9,44,55,66]
print("排序前",c,id(c))
c.sort()
print("排序后",c,id(c))
c.sort(reverse=True)
print("排序后",c,id(c))

print("排序前",c,id(c))
n = sorted(c)
print("排序后",n,id(n))

# 列表生成式
c = [i for i in range(0,11)]
print(c)

c = [i**i for i in range(0,11)]
print(c)

# 2 4 6 8 10
c = [i*2 for i in range(1,6)]
print(c)

# ### 字典
s = {"zhaokai":100,"zhaoli":99}
print(type(s))

v = dict(name="zhaokai",age=27,hometown = "fangshan")
print(type(v))

# 值的获取
print(v["name"])

print(v.get("name"))

# key 判断
print("name" in v)

del v["hometown"]

v["hobby"] = "dog"
print(v)
v["hobby"] = "money"
print(v)

keys =v.keys()
print(list(keys))

val = v.values()
print(val)
print(list(val))

item = v.items()
print(item)
print(list(item))

# 字典的遍历
for item in v:
    print(item,v.get(item))

# 字典生成式
items = ["fruits","books","others"]
prices = [96,97,99]

a = {item.upper():price for item,price in zip(items,prices)}


# 元组的创建
a = ("python",2,33)
print(a,type(a))
b = tuple(("zhaokai",99,22))
print(b,type(b))
# 元组中只有一个元素 必须加逗号 b =
# 遍历
for i in a:
    print(i)

# 集合
s = {2,3,4,5,6,8,6,6,6}
t = set(range(9))
print(type(s),type(t))
a = set(a)
b= set(b)

print(6 in s)

a.add(80)
print(a)
a.update([20,30,40])
print(a)

print(a == b)
print(a != b)

print(s.issubset(t))
print(t.issubset(s))

print(t.issuperset(s))

print(s.isdisjoint(t))

# 集合的数学操作
s.intersection(t)
s&t

s.union(t)
s | t


s = {2,3,4,5,6,8,6,6,6,99,88}
t.difference(s)
t-s

s.symmetric_difference(t)
s^t

# 集合生成式
s = {i*i for i in range(1,11)}


# ## 字符串
# 定义
a = "python，zhaokai"

# 查询
print(a.index("o"))
print(a.find("o"))

print(a.rindex("o"))
print(a.rfind("o"))


# 大小写转换
b = a.upper()
print(b)

c = b.lower()
print(c)

a = "PythOn，zhaOkAi"

b = a.swapcase()
print(b)

c = c.title()
print(c)

b = a.capitalize()
print(b)

#  内容对其
# 剧中
a = a.center(20,"+")
print(a)
# 左右对齐
print(a.ljust(30,"*"))
print(a.rjust(30,"*"))

# 右对齐  0填充
print(a.zfill(30))

# 字符串的分割
a = "hello , PythOn ,zhaO kAi"
lst = a.split()
print(lst)

lst = a.split(sep=",")
lst1 = a.split(sep=",",maxsplit=2)

# 判断 字符串的方法

# 字符串替换  replace()
# 字符串的合并 join()
a = "hello,python,zhaOkAi"
print(a.replace("python", "java"))

b = "hello,python,python,python,python"
print(b.replace("python", "java",2))

lst = ["zhaokai","zhaoli","mama","baba"]
print("+".join(lst))

# 比较












