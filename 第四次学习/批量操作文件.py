# -*- coding:utf-8 -*-
# created by zhoa kai

# f = open("name_list","a")
# f.write("张三5\n")
# f.write("李四3\n")
# f.write("金大王\n")
# f.write("Alex\n")
# f.write("he he\n")
# f.write("jack\n")
#
#
# f.close()
#
# f = open("name_list",mode="r")
# print(f.readline()) # 读一行
# print('------------')
# print(f.read()) # 读出所有内容

# f = open("name_list",mode="w")
# f.write("张三\n")
# f.write("李四\n")
# f.write("金大王\n")
# f.write("Alex\n")
# f.write("zhaokai\n")
# f.close()

# # 嫩模联系方式
# f = open("嫩模联系方式")
# print(f.readlines())
# for line in f:
#     print(line)
#     line=line.split()
#     height = int (line[3])
#     weight = int(line[4])
#     if height >=170 and weight <=50:
#         print(line)

# f = open("nmlxfs",encoding="gbk")
# print(f .read())


import sys
print(sys.argv)
old_str = sys.argv[1]
new_str = sys.argv[2]
filename = sys.argv[3]
#  1.load into ram
f = open(filename,"r+")
data = f.read()
# 2.count and replace
old_str_count = data.count(old_str)
new_data = data.replace(old_str,new_str)
# 3. clear old filename
f.seek(0)
f.truncate()
# 4.save new data into file
f.write(new_data)
print(f"成功替换字符{old_str}to{new_str}，共{old_str_count}")
