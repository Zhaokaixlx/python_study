#!/usr/bin/env python # -*- coding: utf-8 -*
# @Author : 赵凯
# @Time : 2022/4/11 15:59
import re
s = "i study study pythonProject3.97 every day"
print("--------match方法，从开始位置进行匹配--------------------")
print(re.match("i",s).group())
print(re.match("\w",s).group())
print(re.match(".",s).group())

print("--------search方法，从任意位置进行匹配，匹配第一个满足条件的--------------------")
print(re.search("study",s).group())
print(re.search("s\w",s).group())

print("--------findall方法，从任意位置进行匹配，匹配多个满足条件的--------------------")
print(re.findall("y",s))
print(re.findall("python",s))
print(re.findall("p\w.+\d",s))
print(re.findall("p.+\d",s))

print("--------sub方法，替换功能--------------------")
print(re.sub("study","like",s))
print(re.sub("s\w+","likes",s))
