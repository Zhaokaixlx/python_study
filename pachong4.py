#!/usr/bin/env python # -*- coding: utf-8 -*
# @Author : 赵凯
# @Time : 2022/4/10 11:25
import requests
from spyder.plugins.projects.utils import cookie

url = "http://www.baidu.com"
resp = requests.get(url)

# 设置响应的编码格式
resp.encoding="utf-8"
# 获取请求后的cookie信息
cookie = resp.cookies
headers = resp.headers
print("响应状态吗：",resp.status_code)
print("请求后的cookie:",cookie)
print("获取请求的网址：",resp.url)

url1 = "https://www.so.com/"
params = {"q":"python"}
resp = requests.get(url1,params=params)
resp.encoding = "utf-8"
print(resp.text)




