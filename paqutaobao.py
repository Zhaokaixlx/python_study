#!/usr/bin/env python # -*- coding: utf-8 -*
# @Author : 赵凯
# @Time : 2022/4/10 18:48
import requests
from bs4 import BeautifulSoup
url = "https://www.taobao.com/index.php"
headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36 Edg/100.0.1185.36"}
resp = requests.get(url,headers=headers)
# print(resp.text)
bs = BeautifulSoup(resp.text,"lxml")

# 查找所有包含a 标签的内容
a_list = bs.find_all("a")
print(len(a_list))
# 输出a标签中href的值
for i in a_list:
    url = i.get("href")
    # print(url)
    if url==None:
        continue
    if url.startswith("http") or url.startswith("https"):
        print(url)


