#!/usr/bin/env python # -*- coding: utf-8 -*
# @Author : 赵凯
# @Time : 2022/4/7 16:32
import urllib.request
import urllib.parse
url = "https://weibo.com/ceair"
# 发送请求
resp = urllib.request.urlopen(url)
html = resp.read().decode("gbk")
print(html)

# post 请求
url1 = "https://www.xslou.com/login.php"
data = {"username":"18600605736","password":"57365736","action":"login"}
resp = urllib.request.urlopen(url1,data = bytes(urllib.parse.urlencode(data),encoding = "utf-8"))
html1 = resp.read().decode("utf-8")
print(html1)

from http import cookiejar









