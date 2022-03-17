# 从文件中读取整个文件的内容
f = open(r"D:\pycharmcode\第十一次学习\a.txt","r+",encoding="utf-8")
# a = f.read()
a= f.read(100)
# 数字可以自己设定
print(a)
f.close()
# 读一行
b = f.readline()
print(b)
f.close()

# 读取所有的行  储存为列表
b = f.readlines()
print(b)

# 文件的指针  f.seek(offset)  offset =0 开头 =1结尾


# 写入
f = open(r"D:\pycharmcode\第十一次学习\b.txt","w+",encoding="utf-8")
f.write("zhaokai niu bi a ")
f.close()


f = open(r"D:\pycharmcode\第十一次学习\b.txt","w+",encoding="utf-8")
f.writelines(b)
f.close()



# 一维数据存储
f = open(r"D:\pycharmcode\第十一次学习\b.csv","w+",encoding="utf-8")
f.write(",".join(b)+"\n")
f.close()

f = open(r"D:\pycharmcode\第十一次学习\b.csv","r",encoding="utf-8")
c = f.read()
c_now = c.split(",")
f.close()
print(c_now)

# 2维数据存储
f = open(r"D:\pycharmcode\第十一次学习\b.csv","r",encoding="utf-8")
ls=[]
for line in f:
    ls.append(line.strip("\n").split(","))
f.close()









