#!/usr/bin/env python # -*- coding: utf-8 -*
# @Author : 赵凯
# @Time : 2022/4/11 16:23
import requests
import re
url = "https://www.qiushibaike.com/video/"
headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36 Edg/100.0.1185.36"}
resp = requests.get(url,headers=headers)
# print(resp.text)
info = re.findall("<source src='(.*)' type='video/mp4' />",resp.text)
print(info)
# 添加https 并且放入一个列表中
lst = []
for i in info:
    lst.append("https:"+i)
print(lst)

# 下载视频到本地文件
count = 0
for i in lst:
    count+=1
    resp = requests.get(i,headers=headers)
    with open("video/"+str(count)+".mp4","wb") as file:
        file.write(resp.content)
print("视频下载完毕")



