#!/usr/bin/env python # -*- coding: utf-8 -*
# @Author : 赵凯
# @Time : 2022/4/10 10:32
import urllib.request
import urllib.error

# url = "http://www.google.com"
url = "http://www.google.cn"  # 正确的网址
try:
    resp = urllib.request.urlopen(url)
    print(resp)
except urllib.error.URLError as e:
    print(e.reason)

url1 = "https://movie.douban.com/"
try:
    resp = urllib.request.urlopen(url1)
except urllib.error.URLError as e:
    print("原因：",e.reason)
    print("响应状态码：",str(e.code))
    print("响应头数据:",e.headers)



