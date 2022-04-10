#!/usr/bin/env python # -*- coding: utf-8 -*
# @Author : 赵凯
# @Time : 2022/4/10 14:56
import requests
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36'}
res=requests.get('https://image.baidu.com/search/acjson?tn=resultjson_com&logid=11301696314706873159&ipn=rj&ct=201326592&is=&fp=result&fr=&word=%E7%BE%8E%E5%A5%B3&cg=girl&queryWord=%E7%BE%8E%E5%A5%B3&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&expermode=&nojc=&isAsync=&pn=90&rn=30&gsm=5a&1645237235440=',headers=headers)
print(res.json())


url1 = "https://www.baidu.com/img/flexible/logo/pc/result.png"
resp = requests.get(url1)
# 存储
with open("logo.png","wb") as file:
    file.write(resp.content)

