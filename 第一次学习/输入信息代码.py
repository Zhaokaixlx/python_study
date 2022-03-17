# 输入被试基本信息
# name=input("name:")
# age=input("age:")
# hometown=input("hometown:")
# hobbie=input("hobbie:")
# print(name)
# print(age)
# print(hometown)
# print(hobbie)
# input 读的值，都是字符串，即使输入数字
n1=input("num 1:")
n2=input("num 2:")
print(type(n1),type(n2))
# 把字符串转化为数字   int()
print(int(n1)*int(n2))
# 把数字转换为字符串    str()
n3=int(n1)*int(n2)
print(n3,type(str(n3)))



