#!/usr/bin/env python # -*- coding: utf-8 -*
# @Author : 赵凯
# @Time : 2022/4/11 17:18
import requests
from pyquery import PyQuery as py
url = "https://www.qidian.com/rank/newfans/"
headers = {"user-agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36 Edg/100.0.1185.36"}
resp = requests.get(url,headers=headers)
# print(resp.text)
# 初始化 pyquery对象
# 这里使用了字符串初始化方式初始化pyquery对象
doc = py(resp.text)


# 找书名
#a_tag = doc("h2 a")
#print(a_tag)
# 放在列表中--列表生成式
names = [a.text for a in doc("h2 a")]
print(names)

# 找作者
authors = doc("p.author a")
print("------++++++++++++")
print(authors)
author_list = []
for i in range(len(authors)):
    if i%2==0:
        author_list.append(authors[i].text)
print(author_list)

for name,author in zip(names,author_list):
    print(name,":",author)
