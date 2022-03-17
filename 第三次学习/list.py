a = ["zhaokai","zhaoli","baba","mama"]
b = ["shushu","shengzi","zhangli","zhangjie","哈哈","爱新觉罗","A"]
# b.extend(a)  # 合并 可以把另外一个列表的值合并进来
# b.insert(2,"duixiang") # 插入，可以插入任何位置
# b.insert(2,["duixiang","laogong","laopuo"])
# print(b)
# print(b[2])
# print(b[2][1])
# b.pop(-2)
# b.clear()  # 清空
# b.index("shengzi")
# b.count("shushu")
# 在不知道一个元素在列表的哪个位置的情况下，如何修改
# 1. 先判断在不在列表里面，item in list
# 2. 取索引，item_index= b.index("zhangjie")
# 3. 去修改，b[item_index] = "zhangli"
# 切片
# b[:2]  # ['shushu', 'shengzi']
# b[1:] # ['shengzi', 'zhangli']
# print(a)
# 步长切片
# b = ["shushu","shengzi","zhangli","zhangjie","zhaokai","zhaoli","baba","mama"]
# b[0:-1:2]  # ['shushu', 'zhangli', 'zhaokai', 'baba']

# for i in enumerate(b):
#     print(i[0],i[1])
# 结果
# 0 shushu
# 1 shengzi
# 2 zhangli
# 3 zhangjie

# 排序
# b.sort()
# print(b)
# 翻转
# b.reverse()
# print(b)

# 班级成绩列表

