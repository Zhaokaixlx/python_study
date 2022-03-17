# {key1:value1,key2:value2}
# 例如  info = {"name":"alex li","age":26}
dic = {"alex":[23,"ceo",66000],"黑姑娘":[24,"co",6000],"赵凯":[27,"ofc",600000]}
# print(dic["赵凯"])

# 字典的增加操作
dic["赵丽"] = [24,"运营",40000]
# print(dic["赵丽"])

# 删除操作
dic.pop("黑姑娘")
print(dic)

# del dic["黑姑娘"]
# print(dic)

# dic.clear()
# print(dic)   # 清空

# 查询
# "赵凯" in dic  返回值 True

# for i in dic:
#     print(i,dic[i])

# 求长度
# len(dic)  返回值 3




