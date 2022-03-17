# abs 取绝对值
# print(abs(-10))

# all
# a = [1,2,3,4,5,6,7,8,9]
# print(all(a))

# any ,任意一个值为真
# a = [1,2,3,4,5,6,7,8,9]
# print(any(a))

# chr  打印阿斯克码
# print(chr(60))

# dict
# print(dict())
# print(dict(name="zhao kai",age = 25,hometown = "fangshanxian"))

# 打印当前内存中所有的变量名 dir
# name = "alex"
# age = 22
# print("__file__")
# print(dir())

# 打印当前作用域中所有的变量名 & 变量值
# name = "zhao kai"
# age = 25
# print(locals())

# map
# l = list(range(1,11))
# print(l)
# def calc(x): # 只能定义一个参数
#     return x**2
# m = map(calc,l) # 并没有执行(迭代器)
# for i in m:
#     print(i)

# max 最大值 min 最小值 sum 求和
# l = list(range(1,100))
# print(max(l))
# print(min(l))
# print(sum(l))

# ord 打印阿斯克码 ascii 对应于10进制的表
# print(ord("<"))

# eunmerate
# for index,val in enumerate(["zhao kai","zhao li","mama","baba"]):
#     print(index+1,val)

# round 四舍五入取值
# print(round(3.1415926))
# print(round(3.1415926,2))
# print(round(3.1415926,4))

# 转化为字符串 str
# a = str(list(range(1,11)))
# print(type(a))
# <class 'str'>

# zip
# a = ["zhao kai","zhao li","ma ma","ba ba"]
# b = [80000000,8000,2000,8000]
# for i in zip(a,b):
#     print(i)
# ('zhao kai', 80000000)
# ('zhao li', 8000)
# ('ma ma', 2000)
# ('ba ba', 8000)

# filter 把列表里面的每一个元素交给第一个参数（函数）运行，若结果为真，则保留这个值
l = list(range(1,26))
def compare(x):
    if x >18:
        return x
print(l)
for i in filter(compare,l):
    print(i)