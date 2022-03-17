# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 21:42:30 2022

@author: zhaokai
"""
import re
import requests
import os
import chardet   #需要导入这个模块，检测编码格式

def dowmloadPic(html, keyword,i):
    encode_type = chardet.detect(html)
    html = html.decode(encode_type['encoding']) #进行相应解码，赋给原标识符（变量）
    pic_url = re.findall('"objURL":"(.*?)",',html,re.S)

    abc=i*1
    print('找到关键词:' + keyword + '的图片，现在开始下载图片...')
    for each in pic_url:
        print('正在下载第' + str(abc) + '张图片，图片地址:' + str(each))
        try:
            pic = requests.get(each, timeout=10)
        except requests.exceptions.ConnectionError:
            print('【错误】当前图片无法下载')
            continue

        dir = r'D:\pycharmcode\12\pazhouy\i' + keyword + '_' + str(abc) + '.jpg'
        if not os.path.exists('D:\image'):
            os.makedirs('D:\image')

        fp = open(dir, 'wb')
        fp.write(pic.content)
        fp.close()
        abc += 1


if __name__ == '__main__':
    #word = input("Input key word: ")
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36'}
    name = "周也"
    num = 0
    x =1


    for i in range(int(x)):
        url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+name+'+&pn='+str(i*30)
        print(url)
        result = requests.get(url,headers=headers)
        dowmloadPic(result.content, name,1)
print("下载完成")



