#!/usr/bin/env python # -*- coding: utf-8 -*
# @Author : 赵凯
# @Time : 2022/4/10 16:41
from bs4 import BeautifulSoup
html = """
<html>
    <head>  
       <title>赵凯你真牛逼</title>
    </head>
    <body>
       <h1 class="info bg" float="left">蓝与紫的霓虹中，浓密的钢铁苍穹下，数据洪流的前端，是科技革命之后的世界，也是现实与虚幻的分界。钢铁与身体，过去与未来。这里，表世界与里世界并存，面前的一切，像是时间之墙近在眼前。黑暗逐渐笼罩。可你要明白啊我的朋友，我们不能用温柔去应对黑暗，要用火</h1>
       <a href= "https://www.baidu.com">冬奥盛会将长久留下什么？3数万名医护八方驰援 同心守“沪”热110地发现援建方舱阳性人员热4人贩子盯上乌克兰女孩 120万一个新2五一放假安排来了！放假调休共5天热5西安一咖啡店称因影响市貌永久停业</a>
       <h2><!--------注释的内容---------></h2>
    </body>
</html>
"""
# bs = BeautifulSoup(html,"html.parser")
bs1 = BeautifulSoup(html,"lxml")
# 获取标签
print(bs1.title)

# 获取属性
print(bs1.h1.attrs)
# 获取单个属性
print(bs1.h1.get("class"))
print(bs1.h1.get("float"))
print(bs1.h1["class"])
print(bs1.a["href"])

# 获取文本
print(bs1.title.text)
print(bs1.a.text)
print(bs1.a.string)

# 获取内容
print("=====",bs1.h2.string) # 获取了标签中注释的内容
print(bs1.h2.text)  # 什么都没有获取 因为h2中没有正儿八经的内容

html1 = """
    <title>赵凯你他娘的真是一个牛逼的人</title>
    <div class="info" float="left">起点签约的黄金步骤. 起点签约并不难，几乎每一个常年在起点混的作者，都会遵循以下步骤达成签约，新人完全可以借鉴，顺利达成签约</div>
    <div class="info" float="right" id="zk">
      <span>好好学习，天天向上</span>
      <a href="https://www.qidian.com/rank/yuepiao/">赵凯啊</a>
    </div> 
"""
bs = BeautifulSoup(html1,"html.parser")
print(bs.title,type(bs.title))

# 查找内容 第一个满足条件的
print(bs.find("div",class_= "info"),type(bs.find("div",class_= "info")))
print("================")
# 得到所有满足条件的
print(bs.find_all("div",class_= "info"))

# 得倒特定属性的内容
print(bs.find_all("div",attrs={"float":"right"}))


print("------css选择器-------")
print(bs.select("zk"))
print(bs.select(".info"))
print(bs.select("div>span"))
print(bs.select("div.info>span"))

for i in bs.select("div.info>span"):
    print(i.text)
