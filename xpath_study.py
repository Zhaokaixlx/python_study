#!/usr/bin/env python # -*- coding: utf-8 -*
# @Author : 赵凯
# @Time : 2022/4/10 16:09
import requests
from lxml import etree
url = "https://www.qidian.com/rank/yuepiao/"
hearders = {"user-agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36 Edg/100.0.1185.36"}

# 发送请求
resp = requests.get(url,headers=hearders)
e = etree.HTML(resp.text) # 将str类型转换为'lxml.etree._Element'类型

print(type(e))
# 小说的书名
names = e.xpath("//div[@class='book-mid-info']/h2/a/text()")
print(names)
# 小说的作者
authors = e.xpath("//p[@class='author']/a[1]/text()")
print(authors)

# 书的简介
s = e.xpath("//p[@class='intro']/text()")
print(s)

# 用函数写一起
for name,author in zip(names,authors):
    print(name,":",author)
