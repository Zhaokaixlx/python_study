# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:20:23 2022

@author: Administrator
"""


"""
网络爬虫
1.urllib: 内置库，不需要额外安装   使用起来不方便
2.requests:不是内置库，需要额外安装  但是使用起来方便

"""
import requests

url = "https://www.baidu.com" # 网址
response = requests.get(url)
print(response)
# 返回了一个response对象
# 1.response.text
print(response.text)
# response.text这种方法是requests库猜测的编码方式，有可能有误
# 1.response.content
# 返回bytes数据
content = response.content.decode("utf-8")


"""
带参数的查找 get请求
"""
import requests
url = "https://www.baidu.com/s"


kw={"wd":"猫"} # 要搜索的关键词

# 伪装浏览器发起搜索请求
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55"}

response = requests.get(url, params=kw,headers=headers)


# 检查访问的真实的url
print(response.url)

# 解码
content = response.content.decode("utf-8")
print(content)





"""
遗留问题
"""
# 打印出来requests库猜测的解码方式
print(response.encoding)
print(response.text)

# 还可以查看请求状态码
response.status_code


"""
requests库来设置 Cookie参数
"""

import requests
url = "https://www.renren.com/972521253/newsfeed/origin"

# 伪装浏览器发起搜索请求
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55"
           ,"Cookie":"xxxxxxxxx"}

response = requests.get(url, headers=headers)


# 解码
content = response.content.decode("utf-8")
print(content)

with open("renren.html","w",encoding="utf-8") as f:
     f.write(content)

    
"""
requests库来模拟用户登录
"""
import requests
url = "https://www.renren.com/plogin.do"
# 也可以 url = "https://www.renren.com"
# 伪装浏览器发起搜索请求
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55"}

data = {"email":"2781162818@qq.com","password":"python12345"}

# 模拟登陆
session = requests.session()
session.post(url,data = data,headers = headers)

# 访问主页  把主页的网址复制过来
response1 =  session.get("http://www.renren.com/personal/946660161")
content = response1.content.decode("utf-8")

# 写出个人主页
with open("renren.html","w",encoding="utf-8") as f:
    f.write(content)


"""
requests库设置代理ip
"""
import requests
url = "https://httpbin.org/get"

proxy = {"http":"41.204.93.54:8080"}
response = requests.get(url,proxies=proxy)

content = response.content.decode("utf-8")
print(content)



"""
爬取豆瓣影评
"""
import requests
url = "https://movie.douban.com/subject/35215390/"

headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55",
           "Referer":"https://movie.douban.com/review/best/"}

response = requests.get(url,headers=headers)
content = response.content.decode("utf-8")
with open("douban.html","w",encoding="utf-8") as f:
    f.write(content)
    

"""
正则表达式
"""
# 单字符匹配规则
import re
text = "python"
# 特点 ：1.  即使错误，不报错 返回None 2.必需从开头进行匹配
# 1. match()
result = re.match("py",text) # 只能匹配某个
print(result)

# group() 方法 展示结果
print(result.group())


"""
2. 点 (.)
匹配任意的某个字符 【无法匹配换行符】
"""
text = "cpython" # 数字、下划线都可以
result = re.match(".",text) # 只能匹配某个
print(result.group())


"""
3. \d
匹配任意的某个数字 【除了数字外都无法匹配,必须从开头开始】
"""
text = "269cpython" 
result = re.match("\d",text) # 只能匹配某个
print(result.group())



"""
4. \D
除了数字外都可以匹配,必须从开头开始
"""
text = "cpython" 
result = re.match("\D",text) # 只能匹配某个
print(result.group())


"""
5. \s  小写
匹配空白字符  \n \t \r 空格
"""
text = "\ncpython" 
result = re.match("\s",text) 
print(result.group())



"""
6. \w  小写
匹配小写的a-z 大写的A-Z  数字和下划线
"""

text = "cpython" 
result = re.match("\w",text) 
print(result.group())


"""
7. \W  大写
匹配除了小写的\w之外的所有字符
"""
text = "Mcpython" 
result = re.match("\W",text) 
print(result.group())



"""
8.[]   组合的方式
只要在中括号内的内容均可匹配
"""
text = "_ Mcpython" 
result = re.match("[_\s]",text) 
print(result.group())





#####  多字符匹配规则
"""
正则表达式  多字符匹配规则
"""
import re

"""
1.※ 号
匹配0个或者多个字符
"""
text = "184-3522-2133"
result = re.match("[-\d]*", text) 
print(result.group()) 


"""
2.+ 号
匹配1个或者多个字符
"""
text = "a184-3522-2133"
result = re.match("[a\d-]+", text) 
print(result.group()) 

"""
3.？ 号
要么匹配0个， 要么匹配1个
"""
text = "a184-3522-2133"
result = re.match("[a\d-]?", text) 
print(result.group()) 

"""
4.{m} :匹配制定个数m

"""
text = "184-3522-2133"
result = re.match("[\d-]{4}", text) 
print(result.group()) 


"""
5.{m,n} :匹配 m-n 个 默认尽可能匹配最多的

"""
text = "184-3522-2133"
result = re.match("[\d-]{4,11}", text) 
print(result.group()) 



#####  匹配规则 的替代方案
import re

"""
1.匹配所有数字
\d ->> [0-9]
"""
text = "184-3522-2133"
result = re.match("[0-9-]*", text) 
print(result.group()) 

"""
2.匹配所有的非数字
\D ->> [^0-9]
"""
text = "oddn184-3522-2133"
result = re.match("[^0-9]*", text) 
print(result.group()) 



"""
3.匹配所有的数字、字母和下划线
\w ->> [0-9a-zA-Z_]
"""
text = "oddn184_3522_2133ZHAOKAI"
result = re.match("[0-9a-zA-Z_]*", text) 
print(result.group()) 

"""
4.匹配所有的非数字、字母和下划线
\W ->> [^0-9a-zA-Z_]
"""
text = "-oddn184_3522_2133ZHAOKAI"
result = re.match("[^0-9a-zA-Z_]*", text) 
print(result.group()) 


"""
5.匹配所有的字符
[\d\D]
[\w\W]
"""
text = "-oddn184_3522_2133ZHAOKAI"
result = re.match("[\d\D]*", text) 
print(result.group()) 

"""
6.特殊规则
.  匹配任意字符
. 在中括号中仅仅代表匹配点
* 匹配 0个或者多个字符
+ 匹配 1个或者多个字符
？ 匹配 0个或者1个字符
^ 取反
[$ |]

"""
import re 
text = "zhaokai_zhaoli--mama--baba//laolao>>taizi>>cisa"
result = re.match(".+", text)
print(result.group())



"""
正则表达式  案例

"""

"""
1. 验证手机号
[11位 数字]
18435222133
1:->>1
2: ->>[3456789]
3-11:->>[0,9]

"""
text = "18435222133"
result = re.match("1[3456789][0-9]{9}", text)
print(result.group())



"""
2. 验证邮箱
...@xxx.com
1365980632@qq.com
... ->> 英文数字、字母、下划线
xxx->>数字、字母【大写不存在】
"""
text = "1365980632@qq.com"
result = re.match("\w+@[0-9a-z]+[.]com", text)
print(result.group())



"""
3. 验证身份证号
18位
前17位 [0-9]
第十八位：[0-9xX]

"""
text = "321282198110275214"

result = re.match("[0-9]{17}[0-9xX]", text)
print(result.group())




"""
再谈特殊字符  补充
re.match()  必须从字符串开头进行匹配
re.search() 从左到右进行字符串的遍历，找到就返回

1. 脱字号 ^
(1)在中括号内表示取反
(2)在中括号外 表示从指定的字符串开始
"""
# 1
import re
text = "13453python"
result = re.search("[^\d]+",text) 
print(result.group())
# 2
import re
text = "abcgshsnpython"
result = re.search("a.*?p",text) # a开始，b结束
print(result.group())

"""
2.$
以....为结尾
"""
text = "1365980632@qq.com"
result = re.match("\w+@[0-9a-z]+[.]com$", text)
print(result.group())

"""
3.|
匹配多个表达式或者字符串
"""
# (1)中括号 里面都是单个字符
# （2）小括号认为是不同的字符串
import re
text = "https"
result = re.search("[http|https|ftp|file]+",text) 
print(result.group())


import re
text = "https"
result = re.search("(http|https|ftp|file)",text) 
print(result.group())




"""
正则表达式  贪婪模式和非贪婪模式

贪婪模式: 正则表达式会尽可能多的匹配字符[默认贪婪模式]

非贪婪模式:尽可能少的匹配字符【？】

"""
import re 
text = "python"
result= re.match("[a-z]+",text)
print(result.group())

import re 
text = "python"
result= re.match("[a-z]+?",text)
print(result.group())

import re 
text = \
    """
<div class="container sub-navigation sub-navigation-articles" style="display:none">
	<div class="row">
		<div class="col nav-sub">
		<ul id="python">
			<li class="cat-item"><a href="/python3/python3-tutorial.html">Python3 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
			<li class="cat-item"><a href="/python/python-tutorial.html">Python2 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
		</ul>
		<ul id="vue">
			<li class="cat-item"><a href="/vue3/vue3-tutorial.html">Vue3 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
			<li class="cat-item"><a href="/vue/vue-tutorial.html">vue2 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
		</ul>

		<ul id="bootstrap">
			<li class="cat-item"><a href="/bootstrap/bootstrap-tutorial.html">Bootstrap3 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
			<li class="cat-item"><a href="/bootstrap4/bootstrap4-tutorial.html">Bootstrap4 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
			<li class="cat-item"><a href="/bootstrap5/bootstrap5-tutorial.html">Bootstrap5 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
			<li class="cat-item"><a href="/bootstrap/bootstrap-v2-tutorial.html">Bootstrap2 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
		</ul>
		</div>
	</div>
</div><!--  内容  -->
"""
result = re.match("\s<div[\d\D]+?>", text)
print(result.group())


"""
正则表达式  转义字符 \
    盘点特殊字符
(.) ->匹配任意字符
(*) ->匹配0个或者多个字符
(+) ->匹配1个或者多个字符
(?) ->要么1个要么0个  非贪婪
($) -> 以....结尾
(|) ->或 通过小括号
(^) ->1.中括号内表示取反 2.中括号外表示以....为开始
"""
import re
text = "3.1415926"
result = re.match("\d[.]\d+",text)
print(result.group())
# 通过\转义
import re
text = "***3.1415926****"
result = re.match("\*+\d\.\d+\*+",text)
print(result.group())


"""
正则表达式  group 分组  
  把匹配到的数据 拿出来
"""
import re
text =  "my email is 1365980632@qq.com and python123@163.com"
result = re.match("[\s\w]+\s(\w+@[0-9a-z]+.com)[\s\w]+\s(\w+@[0-9a-z]+.com)", text)

result = re.search("\s(\w+@[0-9a-z]+.com)[\s\w]+\s(\w+@[0-9a-z]+.com)", text)
print(result.group())
print(result.group(1))
print(result.group(2))

"""
正则表达式  的常用函数
1.re.match():从开头从左到右进行匹配【开头不满足即失败】
2.re.seach():在整个字符串中查找，返回第一个被找到的字符串【只返回第一个】
3.re.findall():在整个字符串中查找所有满足条件的字符串【返回结果为列表】
4.re.sub:替换字符串 【匹配出来的字符串进行人为替换】
"""
import re
text = "my email is 1365980632@qq.com and python123@163.com"
result = re.findall("\s(\w+@[0-9a-z]+\.com)",text)
print(result)

import re
text = "my email is 1365980632@qq.com and python123@163.com"
result = re.sub("\s(\w+@[0-9a-z]+\.com)"," it was wrong",text)
print(result)

"""
5. re.split():主要用来分割字符串 

"""
import re
text = "my email is 1365980632@qq.com and python123@163.com"
result = re.split("[^\w]",text)
print(result)

"""
6. re.compile():进行编译

"""
r = re.compile(r"""
               \s  #邮箱前的空格
               (\w+ #邮箱的第一部分
                @
                [0-9a-z]+ #邮箱的第二部分
                \.com)
               """,  re.VERBOSE)
result = re.findall(r, text)
print(result)




"""
正则表达式的应用
 
  
"""
# 导入数据
text = """

<!Doctype html>
<html>
<head>
	<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1.0" />
	<title>Python3 教程 | 菜鸟教程</title>

  <meta name='robots' content='max-image-preview:large' />
<style id='global-styles-inline-css' type='text/css'>
body{--wp--preset--color--black: #000000;--wp--preset--color--cyan-bluish-gray: #abb8c3;--wp--preset--color--white: #ffffff;--wp--preset--color--pale-pink: #f78da7;--wp--preset--color--vivid-red: #cf2e2e;--wp--preset--color--luminous-vivid-orange: #ff6900;--wp--preset--color--luminous-vivid-amber: #fcb900;--wp--preset--color--light-green-cyan: #7bdcb5;--wp--preset--color--vivid-green-cyan: #00d084;--wp--preset--color--pale-cyan-blue: #8ed1fc;--wp--preset--color--vivid-cyan-blue: #0693e3;--wp--preset--color--vivid-purple: #9b51e0;--wp--preset--gradient--vivid-cyan-blue-to-vivid-purple: linear-gradient(135deg,rgba(6,147,227,1) 0%,rgb(155,81,224) 100%);--wp--preset--gradient--light-green-cyan-to-vivid-green-cyan: linear-gradient(135deg,rgb(122,220,180) 0%,rgb(0,208,130) 100%);--wp--preset--gradient--luminous-vivid-amber-to-luminous-vivid-orange: linear-gradient(135deg,rgba(252,185,0,1) 0%,rgba(255,105,0,1) 100%);--wp--preset--gradient--luminous-vivid-orange-to-vivid-red: linear-gradient(135deg,rgba(255,105,0,1) 0%,rgb(207,46,46) 100%);--wp--preset--gradient--very-light-gray-to-cyan-bluish-gray: linear-gradient(135deg,rgb(238,238,238) 0%,rgb(169,184,195) 100%);--wp--preset--gradient--cool-to-warm-spectrum: linear-gradient(135deg,rgb(74,234,220) 0%,rgb(151,120,209) 20%,rgb(207,42,186) 40%,rgb(238,44,130) 60%,rgb(251,105,98) 80%,rgb(254,248,76) 100%);--wp--preset--gradient--blush-light-purple: linear-gradient(135deg,rgb(255,206,236) 0%,rgb(152,150,240) 100%);--wp--preset--gradient--blush-bordeaux: linear-gradient(135deg,rgb(254,205,165) 0%,rgb(254,45,45) 50%,rgb(107,0,62) 100%);--wp--preset--gradient--luminous-dusk: linear-gradient(135deg,rgb(255,203,112) 0%,rgb(199,81,192) 50%,rgb(65,88,208) 100%);--wp--preset--gradient--pale-ocean: linear-gradient(135deg,rgb(255,245,203) 0%,rgb(182,227,212) 50%,rgb(51,167,181) 100%);--wp--preset--gradient--electric-grass: linear-gradient(135deg,rgb(202,248,128) 0%,rgb(113,206,126) 100%);--wp--preset--gradient--midnight: linear-gradient(135deg,rgb(2,3,129) 0%,rgb(40,116,252) 100%);--wp--preset--duotone--dark-grayscale: url('#wp-duotone-dark-grayscale');--wp--preset--duotone--grayscale: url('#wp-duotone-grayscale');--wp--preset--duotone--purple-yellow: url('#wp-duotone-purple-yellow');--wp--preset--duotone--blue-red: url('#wp-duotone-blue-red');--wp--preset--duotone--midnight: url('#wp-duotone-midnight');--wp--preset--duotone--magenta-yellow: url('#wp-duotone-magenta-yellow');--wp--preset--duotone--purple-green: url('#wp-duotone-purple-green');--wp--preset--duotone--blue-orange: url('#wp-duotone-blue-orange');--wp--preset--font-size--small: 13px;--wp--preset--font-size--medium: 20px;--wp--preset--font-size--large: 36px;--wp--preset--font-size--x-large: 42px;}.has-black-color{color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-color{color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-color{color: var(--wp--preset--color--white) !important;}.has-pale-pink-color{color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-color{color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-color{color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-color{color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-color{color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-color{color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-color{color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-color{color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-color{color: var(--wp--preset--color--vivid-purple) !important;}.has-black-background-color{background-color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-background-color{background-color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-background-color{background-color: var(--wp--preset--color--white) !important;}.has-pale-pink-background-color{background-color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-background-color{background-color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-background-color{background-color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-background-color{background-color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-background-color{background-color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-background-color{background-color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-background-color{background-color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-background-color{background-color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-background-color{background-color: var(--wp--preset--color--vivid-purple) !important;}.has-black-border-color{border-color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-border-color{border-color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-border-color{border-color: var(--wp--preset--color--white) !important;}.has-pale-pink-border-color{border-color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-border-color{border-color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-border-color{border-color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-border-color{border-color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-border-color{border-color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-border-color{border-color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-border-color{border-color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-border-color{border-color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-border-color{border-color: var(--wp--preset--color--vivid-purple) !important;}.has-vivid-cyan-blue-to-vivid-purple-gradient-background{background: var(--wp--preset--gradient--vivid-cyan-blue-to-vivid-purple) !important;}.has-light-green-cyan-to-vivid-green-cyan-gradient-background{background: var(--wp--preset--gradient--light-green-cyan-to-vivid-green-cyan) !important;}.has-luminous-vivid-amber-to-luminous-vivid-orange-gradient-background{background: var(--wp--preset--gradient--luminous-vivid-amber-to-luminous-vivid-orange) !important;}.has-luminous-vivid-orange-to-vivid-red-gradient-background{background: var(--wp--preset--gradient--luminous-vivid-orange-to-vivid-red) !important;}.has-very-light-gray-to-cyan-bluish-gray-gradient-background{background: var(--wp--preset--gradient--very-light-gray-to-cyan-bluish-gray) !important;}.has-cool-to-warm-spectrum-gradient-background{background: var(--wp--preset--gradient--cool-to-warm-spectrum) !important;}.has-blush-light-purple-gradient-background{background: var(--wp--preset--gradient--blush-light-purple) !important;}.has-blush-bordeaux-gradient-background{background: var(--wp--preset--gradient--blush-bordeaux) !important;}.has-luminous-dusk-gradient-background{background: var(--wp--preset--gradient--luminous-dusk) !important;}.has-pale-ocean-gradient-background{background: var(--wp--preset--gradient--pale-ocean) !important;}.has-electric-grass-gradient-background{background: var(--wp--preset--gradient--electric-grass) !important;}.has-midnight-gradient-background{background: var(--wp--preset--gradient--midnight) !important;}.has-small-font-size{font-size: var(--wp--preset--font-size--small) !important;}.has-medium-font-size{font-size: var(--wp--preset--font-size--medium) !important;}.has-large-font-size{font-size: var(--wp--preset--font-size--large) !important;}.has-x-large-font-size{font-size: var(--wp--preset--font-size--x-large) !important;}
</style>
<link rel="canonical" href="http://www.runoob.com/python3/python3-tutorial.html" />
<meta name="keywords" content="Python3 教程,Python,python3,python教程">
<meta name="description" content="Python 3 教程    Python 的 3.0 版本，常被称为 Python 3000，或简称 Py3k。相对于 Python 的早期版本，这是一个较大的升级。为了不带入过多的累赘，Python 3.0 在设计的时候没有考虑向下兼容。 Python 介绍及安装教程我们在Python 2.X 版本的教程中已有介绍，这里就不再赘述。 你也可以点击  Python2.x与3​​.x版本区别 来查看两者的不同。 本教程主要针对 Pyth..">
		
	<link rel="shortcut icon" href="https://static.runoob.com/images/favicon.ico" mce_href="//static.runoob.com/images/favicon.ico" type="image/x-icon" >
	<link rel="stylesheet" href="/wp-content/themes/runoob/style.css?v=1.165" type="text/css" media="all" />	
	<link rel="stylesheet" href="https://cdn.staticfile.org/font-awesome/4.7.0/css/font-awesome.min.css" media="all" />	
  <!--[if gte IE 9]><!-->
  <script src="https://cdn.staticfile.org/jquery/2.0.3/jquery.min.js"></script>
  <!--<![endif]-->
  <!--[if lt IE 9]>
     <script src="https://cdn.staticfile.org/jquery/1.9.1/jquery.min.js"></script>
     <script src="https://cdn.staticfile.org/html5shiv/r29/html5.min.js"></script>
  <![endif]-->
  <link rel="apple-touch-icon" href="https://static.runoob.com/images/icon/mobile-icon.png"/>
  <meta name="apple-mobile-web-app-title" content="菜鸟教程">
</head>
<body>

<!--  头部 -->
<div class="container logo-search">

  <div class="col search row-search-mobile">
    <form action="index.php">
      <input class="placeholder" placeholder="搜索……" name="s" autocomplete="off">
      
    </form>
  </div>

  <div class="row">
    <div class="col logo">
      <h1><a href="/">菜鸟教程 -- 学的不仅是技术，更是梦想！</a></h1>
    </div>
        <div class="col right-list"> 
    <button class="btn btn-responsive-nav btn-inverse" data-toggle="collapse" data-target=".nav-main-collapse" id="pull" style=""> <i class="fa fa-navicon"></i> </button>
    </div>
        
    <div class="col search search-desktop last">
      <div class="search-input" >
      <form action="//www.runoob.com/" target="_blank">
        <input class="placeholder" id="s" name="s" placeholder="搜索……"  autocomplete="off" style="height: 44px;">
      </form>
      
      </div>
    </div>
  </div>
</div>


<!-- 导航栏 -->
<!-- 导航栏 -->
<div class="container navigation">
	<div class="row">
		<div class="col nav">
			<ul class="pc-nav" id="runoob-detail-nav">
				<li><a href="//www.runoob.com/">首页</a></li>
				<li><a href="/html/html-tutorial.html">HTML</a></li>
				<li><a href="/css/css-tutorial.html">CSS</a></li>
				<li><a href="/js/js-tutorial.html">JavaScript</a></li>
				
				<li><a href="javascript:void(0);" data-id="vue">Vue</a></li>
				<li><a href="javascript:void(0);" data-id="bootstrap">Bootstrap</a></li>
				<li><a href="/nodejs/nodejs-tutorial.html">NodeJS</a></li>
				<li><a href="/jquery/jquery-tutorial.html">jQuery</a></li>
				<li><a href="javascript:void(0);" data-id="python">Python</a>
				
				</li>
				<li><a href="/java/java-tutorial.html">Java</a></li>
				<li><a href="/cprogramming/c-tutorial.html">C</a></li>
				<li><a href="/cplusplus/cpp-tutorial.html">C++</a></li>
				<li><a href="/csharp/csharp-tutorial.html">C#</a></li>
				<li><a href="/go/go-tutorial.html">Go</a></li>
				<li><a href="/sql/sql-tutorial.html">SQL</a></li>
				<li><a href="/linux/linux-tutorial.html">Linux</a></li>
				<li><a href="/browser-history">本地书签</a></li>
				<!--
			
				<li><a href="/w3cnote/knowledge-start.html" style="font-weight: bold;" onclick="_hmt.push(['_trackEvent', '星球', 'click', 'start'])" title="我的圈子">我的圈子</a></li>				
				<li><a href="javascript:;" class="runoob-pop">登录</a></li>
				-->
      		</ul>
			<ul class="mobile-nav">
				<li><a href="//www.runoob.com/">首页</a></li>
				<li><a href="/html/html-tutorial.html">HTML</a></li>
				<li><a href="/css/css-tutorial.html">CSS</a></li>
				<li><a href="/js/js-tutorial.html">JS</a></li>
				<li><a href="/browser-history">本地书签</a></li>
				<li><a href="javascript:void(0)" class="search-reveal">Search</a> </li>
			</ul>
			
		</div>
	</div>
</div>

<div class="container sub-navigation sub-navigation-articles" style="display:none">
	<div class="row">
		<div class="col nav-sub">
		<ul id="python">
			<li class="cat-item"><a href="/python3/python3-tutorial.html">Python3 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
			<li class="cat-item"><a href="/python/python-tutorial.html">Python2 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
		</ul>
		<ul id="vue">
			<li class="cat-item"><a href="/vue3/vue3-tutorial.html">Vue3 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
			<li class="cat-item"><a href="/vue/vue-tutorial.html">vue2 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
		</ul>

		<ul id="bootstrap">
			<li class="cat-item"><a href="/bootstrap/bootstrap-tutorial.html">Bootstrap3 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
			<li class="cat-item"><a href="/bootstrap4/bootstrap4-tutorial.html">Bootstrap4 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
			<li class="cat-item"><a href="/bootstrap5/bootstrap5-tutorial.html">Bootstrap5 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
			<li class="cat-item"><a href="/bootstrap/bootstrap-v2-tutorial.html">Bootstrap2 教程 <i class="fa fa-external-link" aria-hidden="true"></i></a></li>
		</ul>
		</div>
	</div>
</div><!--  内容  -->
<div class="container main">
	<!-- 中间 -->
	<div class="row">
	
<div class="runoob-col-md2">
<div class="left-column">
	<div class="tab" style="overflow: hidden;text-overflow: ellipsis;white-space: nowrap;">
	<i class="fa fa-list" aria-hidden="true"></i> 
	<span>Python 3 教程</span>
	<a data-cate="124" href="javascript:void(0);" title="夜间模式"  id="moon"><i class="fa fa-moon-o" aria-hidden="true" style="line-height: 28px;font-size: 1.4em;margin: 4px 6px 0;"></i></a>
	<a data-cate="124" style="display:none;" href="javascript:void(0);" title="日间模式"  id="sun" ><i class="fa fa-sun-o" aria-hidden="true" style="line-height: 28px;font-size: 1.4em;margin: 4px 6px 0;line-height: 28px;
}"></i></a>
	</div>
	<div class="sidebar-box gallery-list">
		<div class="design" id="leftcolumn">
						<a target="_top" title="Python3 教程"  href="/python3/python3-tutorial.html" >
			Python3 教程			</a>
			<a target="_top" title="Python3 简介" href="python3-intro.html" > Python3 简介 </a>
<a target="_top" title="Python3 环境搭建" href="python3-install.html" > Python3 环境搭建</a>
<a target="_top" title="Python3 VScode" href="python-vscode-setup.html"> Python3 VScode </a>
<a target="_top" title="Python3 基础语法" href="python3-basic-syntax.html" > Python3 基础语法 </a>
<a target="_top" title="Python3 基本数据类型" href="python3-data-type.html" > Python3 基本数据类型 </a>
<a target="_top" title="Python3  数据类型转换" href="python3-type-conversion.html" > Python3 数据类型转换 </a>
<a target="_top" title="Python3 推导式" href="python-comprehensions.html" > Python3 推导式 </a>			<a target="_top" title="Python3 解释器"  href="/python3/python3-interpreter.html" >
			Python3 解释器			</a>
						<a target="_top" title="Python3 注释"  href="/python3/python3-comment.html" >
			Python3 注释			</a>
			<a target="_top" title="Python3 运算符" href="python3-basic-operators.html"> Python3 运算符 </a>			<a target="_top" title="Python3 数字(Number)"  href="/python3/python3-number.html" >
			Python3 数字(Number)			</a>
						<a target="_top" title="Python3 字符串"  href="/python3/python3-string.html" >
			Python3 字符串			</a>
						<a target="_top" title="Python3 列表"  href="/python3/python3-list.html" >
			Python3 列表			</a>
			<a target="_top" title="Python3 元组" href="python3-tuple.html"> Python3 元组 </a>
<a target="_top" title="Python3 字典" href="python3-dictionary.html"> Python3 字典</a>
<a target="_top" title="Python3 集合" href="python3-set.html"> Python3 集合</a>			<a target="_top" title="Python3 编程第一步"  href="/python3/python3-step1.html" >
			Python3 编程第一步			</a>
						<a target="_top" title="Python3 条件控制"  href="/python3/python3-conditional-statements.html" >
			Python3 条件控制			</a>
						<a target="_top" title="Python3 循环语句"  href="/python3/python3-loop.html" >
			Python3 循环语句			</a>
			<a target="_top" title="Python3 迭代器与生成器" href="python3-iterator-generator.html"> Python3 迭代器与生成器</a>			<a target="_top" title="Python3 函数"  href="/python3/python3-function.html" >
			Python3 函数			</a>
						<a target="_top" title="Python3 数据结构"  href="/python3/python3-data-structure.html" >
			Python3 数据结构			</a>
						<a target="_top" title="Python3 模块"  href="/python3/python3-module.html" >
			Python3 模块			</a>
						<a target="_top" title="Python3 输入和输出"  href="/python3/python3-inputoutput.html" >
			Python3 输入和输出			</a>
			<a target="_top" title="Python3 File" href="python3-file-methods.html"> Python3 File </a>
<a target="_top" title="Python3 OS" href="python3-os-file-methods.html"> Python3 OS </a>			<a target="_top" title="Python3 错误和异常"  href="/python3/python3-errors-execptions.html" >
			Python3 错误和异常			</a>
						<a target="_top" title="Python3 面向对象"  href="/python3/python3-class.html" >
			Python3 面向对象			</a>
			<a target="_top" title=" Python3 命名空间/作用域" href="/python3/python3-namespace-scope.html"> Python3 命名空间/作用域</a>			<a target="_top" title="Python3 标准库概览"  href="/python3/python3-stdlib.html" >
			Python3 标准库概览			</a>
						<a target="_top" title="Python3 实例"  href="/python3/python3-examples.html" >
			Python3 实例			</a>
			<a target="_blank" title="Python 测验" href="/quiz/python-quiz.html"> Python 测验 </a>
<br><h2 class="left"><span class="left_h2">Python3</span> 高级教程</h2>			<a target="_top" title="Python3 正则表达式"  href="/python3/python3-reg-expressions.html" >
			Python3 正则表达式			</a>
						<a target="_top" title="Python3 CGI编程"  href="/python3/python3-cgi-programming.html" >
			Python3 CGI编程			</a>
			<a target="_top" title="Python MySQL - mysql-connector 驱动" href="python-mysql-connector.html">Python3 MySQL(mysql-connector)</a>			<a target="_top" title="Python3 MySQL 数据库连接 &#8211; PyMySQL 驱动"  href="/python3/python3-mysql.html" >
			Python3 MySQL(PyMySQL)			</a>
						<a target="_top" title="Python3 网络编程"  href="/python3/python3-socket.html" >
			Python3 网络编程			</a>
						<a target="_top" title="Python3 SMTP发送邮件"  href="/python3/python3-smtp.html" >
			Python3 SMTP发送邮件			</a>
						<a target="_top" title="Python3 多线程"  href="/python3/python3-multithreading.html" >
			Python3 多线程			</a>
						<a target="_top" title="Python3 XML 解析"  href="/python3/python3-xml-processing.html" >
			Python3 XML 解析			</a>
						<a target="_top" title="Python3 JSON 数据解析"  href="/python3/python3-json.html" >
			Python3 JSON			</a>
						<a target="_top" title="Python3 日期和时间"  href="/python3/python3-date-time.html" >
			Python3 日期和时间			</a>
						<a target="_top" title="Python3 内置函数"  href="/python3/python3-built-in-functions.html" >
			Python3 内置函数			</a>
						<a target="_top" title="Python MongoDB"  href="/python3/python-mongodb.html" >
			Python3 MongoDB			</a>
			<a target="_top" title="Python3 urllib" href="python-urllib.html"> Python3 urllib</a>			<a target="_top" title="Python uWSGI  安装配置"  href="/python3/python-uwsgi.html" >
			Python uWSGI  安装配置			</a>
						<a target="_top" title="Python3 pip"  href="/python3/python3-pip.html" >
			Python3 pip			</a>
						<a target="_top" title="Python 移除列表中重复的元素"  href="/python3/python-remove-duplicate-from-list.html" >
			Python 移除列表中重复的元素			</a>
				
		</div>
	</div>	
</div>
</div>	<div class="col middle-column">
		
	
	<div class="article">
			<div class="article-heading-ad" style="display: none;">
		
		</div>
		<div class="previous-next-links">
			<div class="previous-design-link"> </div>
			<div class="next-design-link"><a href="http://www.runoob.com/python3/python3-interpreter.html" rel="next"> Python3 解释器</a> <i style="font-size:16px;" class="fa fa-arrow-right" aria-hidden="true"></i></div>
		</div>
		<div class="article-body">
		
			<div class="article-intro" id="content">
			
			<h1>Python 3 教程</h1>
<div class="tutintro"> 
<img src="/wp-content/uploads/2014/05/python3.png" alt="python3" width="150" height="81"/>
<p>Python 的 3.0 版本，常被称为 Python 3000，或简称 Py3k。相对于 Python 的早期版本，这是一个较大的升级。为了不带入过多的累赘，Python 3.0 在设计的时候没有考虑向下兼容。</p>
<p>Python 介绍及安装教程我们在<a href="/python/python-tutorial.html" target="_blank" rel="noopener noreferrer">Python 2.X 版本的教程</a>中已有介绍，这里就不再赘述。<a/>
<p>你也可以点击 <a target="_top" href="python-2x-3x.html" rel="noopener noreferrer"> Python2.x与3​​.x版本区别 </a>来查看两者的不同。</p>
<p>本教程主要针对 Python 3.x 版本的学习，如果你使用的是 Python 2.x 版本请移步至 <a href="/python/python-tutorial.html" target="_blank" rel="noopener noreferrer">Python 2.X 版本的教程</a>。
<p><strong>官方宣布，2020 年 1 月 1 日， 停止 Python 2 的更新。</strong></p>
</div>
<hr>
<h2>查看 Python 版本</h2>
<p>我们可以在命令窗口(Windows 使用 win+R 调出 cmd 运行框)使用以下命令查看我们使用的 Python 版本：</p>
<pre>
python -V
或
python --version
</pre>
<p>以上命令执行结果如下：</p>
<pre>
Python 3.3.2
</pre>
<p>你也可以进入Python的交互式编程模式，查看版本：</p>
<pre>
Python 3.3.2 (v3.3.2:d047928ae3f6, May 16 2013, 00:03:43) [MSC v.1600 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
&gt;&gt;&gt; 
</pre>
<hr>
<h2>第一个Python3.x程序</h2>
<p>对于大多数程序语言，第一个入门编程代码便是"Hello World！"，以下代码为使用Python输出"Hello World！"：</p>
<div class="example"> 
<h2 class="example">实例(Python 3.0+)</h2> 
<div class="example_code">
<div class="hl-main"><span class="hl-comment">#!/usr/bin/python3</span><span class="hl-code">
 
</span><span class="hl-identifier">print</span><span class="hl-brackets">(</span><span class="hl-quotes">&quot;</span><span class="hl-string">Hello, World!</span><span class="hl-quotes">&quot;</span><span class="hl-brackets">)</span></div>
</div><br>
<a target="_blank" href="/try/runcode.php?filename=HelloWorld&type=python3" class="showbtn" rel="noopener noreferrer">运行实例 »</a>
</div>
<p>你可以将以上代码保存在 hello.py 文件中并使用 python 命令执行该脚本文件。</p>
<pre>
$ python3 hello.py
</pre>
<p>以上命令输出结果为：</p>
<pre>
Hello, World!
</pre>
<hr><h2>相关内容：</h2>
<p> <a target="_blank" href="/manual/pythontutorial3/docs/html/" rel="noopener noreferrer">Python 3.6.3 中文手册</a></p>
<p><a href="/python/python-tutorial.html" target="_blank" rel="noopener noreferrer">Python 2.X 版本的教程</a></p>			<!-- 其他扩展 -->
						
			</div>
			
		</div>
		
		<div class="previous-next-links">
			<div class="previous-design-link"> </div>
			<div class="next-design-link"><a href="http://www.runoob.com/python3/python3-interpreter.html" rel="next"> Python3 解释器</a> <i style="font-size:16px;" class="fa fa-arrow-right" aria-hidden="true"></i></div>
		</div>
		<!-- 笔记列表 -->
		<style>
.wrapper {
  /*text-transform: uppercase; */
  background: #ececec;
  color: #555;
  cursor: help;
  font-family: "Gill Sans", Impact, sans-serif;
  font-size: 20px;
  position: relative;
  text-align: center;
  width: 200px;
  -webkit-transform: translateZ(0); /* webkit flicker fix */
  -webkit-font-smoothing: antialiased; /* webkit text rendering fix */
}

.wrapper .tooltip {
  white-space: nowrap;
  font-size: 14px;
  text-align: left;
  background: #96b97d;
  bottom: 100%;
  color: #fff;
  display: block;
  left: -25px;
  margin-bottom: 15px;
  opacity: 0;
  padding: 14px;
  pointer-events: none;
  position: absolute;
  
  -webkit-transform: translateY(10px);
     -moz-transform: translateY(10px);
      -ms-transform: translateY(10px);
       -o-transform: translateY(10px);
          transform: translateY(10px);
  -webkit-transition: all .25s ease-out;
     -moz-transition: all .25s ease-out;
      -ms-transition: all .25s ease-out;
       -o-transition: all .25s ease-out;
          transition: all .25s ease-out;
  -webkit-box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.28);
     -moz-box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.28);
      -ms-box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.28);
       -o-box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.28);
          box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.28);
}
.tooltip a {
	color:#fff;
}
/* This bridges the gap so you can mouse into the tooltip without it disappearing */
.wrapper .tooltip:before {
  bottom: -20px;
  content: " ";
  display: block;
  height: 20px;
  left: 0;
  position: absolute;
  width: 100%;
}  

/* CSS Triangles - see Trevor's post */
.wrapper .tooltip:after {
  border-left: solid transparent 10px;
  border-right: solid transparent 10px;
  border-top: solid #96b97d 10px;
  bottom: -10px;
  content: " ";
  height: 0;
  left: 20%;
  margin-left: -13px;
  position: absolute;
  width: 0;
}
.wrapper .tooltip1 {
	margin-left: 50px;
	padding-top: 0px;
}
/*
.wrapper:hover .tooltip {
  opacity: 1;
  pointer-events: auto;
  -webkit-transform: translateY(0px);
     -moz-transform: translateY(0px);
      -ms-transform: translateY(0px);
       -o-transform: translateY(0px);
          transform: translateY(0px);
}
*/
/* IE can just show/hide with no transition */
.lte8 .wrapper .tooltip {
  display: none;
}

.lte8 .wrapper:hover .tooltip {
  display: block;
}

</style>

<link rel="stylesheet" href="https://static.runoob.com/assets/upvotejs/dist/upvotejs/upvotejs.css">
<script src="https://static.runoob.com/assets/upvotejs/dist/upvotejs/upvotejs.vanilla.js"></script>
<script src="https://static.runoob.com/assets/upvotejs/dist/upvotejs/upvotejs.jquery.js"></script>
<div class="title" id="comments">
	<h2 class="">
    <div class="altblock">
				<i style="font-size:28px;margin-top: 8px;" class="fa fa-plus-square" aria-hidden="true"></i>
		    </div>
    <span class="mw-headline" id="qa_headline">4  篇笔记</span>
	<span class="mw-headline" id="user_add_note" style="float:right;line-height: 62px;padding-right: 14px;"><i class="fa fa-pencil-square-o" aria-hidden="true"></i>  写笔记</span>
    </h2>
</div>

<div id="postcomments"  style="display:none;" >
	<ol class="commentlist">
		<li class="comment even thread-even depth-1" id="comment-21438"><span class="comt-f">#0</span><div class="comt-avatar wrapper"><i style="font-size:36px;" class="fa fa-user-circle" aria-hidden="true"></i><div class="tooltip"><p><i class="fa fa-user" aria-hidden="true"></i>&nbsp;&nbsp;&nbsp;helloworld</p><p><i class="fa fa-envelope" aria-hidden="true"></i>&nbsp;&nbsp;229***137@qq.com</p></div><div id="runoobvote-id-21438" data-commid = "21438" class="upvotejs"><a class="upvote"></a> <span class="count">818</span></div></div><div class="comt-main" id="div-comment-21438"><p data-title="Python #!/usr/bin/python 解析" data-commid="22375, 26799">关于实例中第一行代码<span class="marked">#!/usr/bin/python3</span> 的理解：</p>

<p>分成两种情况：</p>
<p><strong>（1）</strong>如果调用python脚本时，使用:</p>

<pre>python script.py </pre>

<p><span class="marked">#!/usr/bin/python</span> 被忽略，等同于注释。</p>

<p><strong>（2）</strong>如果调用python脚本时，使用:</p>

<pre>./script.py </pre>

<p><span class="marked">#!/usr/bin/python</span> 指定解释器的路径。</p>

<div class="comt-meta wrapper"><span class="comt-author"><a target="_blank" href="/note/21438">helloworld</a><div class="tooltip tooltip1"><p><i class="fa fa-user" aria-hidden="true"></i>&nbsp;&nbsp;&nbsp;helloworld</p><p><i class="fa fa-envelope" aria-hidden="true"></i>&nbsp;&nbsp;229***137@qq.com</p></div></span>4年前 (2017-12-10)</div></div></li><!-- #comment-## -->
<li class="comment odd alt thread-odd thread-alt depth-1" id="comment-22375"><span class="comt-f">#0</span><div class="comt-avatar wrapper"><i style="font-size:36px;" class="fa fa-user-circle" aria-hidden="true"></i><div class="tooltip"><p><i class="fa fa-user" aria-hidden="true"></i>&nbsp;&nbsp;&nbsp;Xander663</p><p><i class="fa fa-envelope" aria-hidden="true"></i>&nbsp;&nbsp;xan***1998@163.com</p></div><div id="runoobvote-id-22375" data-commid = "22375" class="upvotejs"><a class="upvote"></a> <span class="count">395</span></div></div><div class="comt-main" id="div-comment-22375"><p>再解释一下第一行代码<code><b>#!/usr/bin/python3</b></code></p><p>这句话仅仅在linux或unix系统下有作用，在windows下无论在代码里加什么都无法直接运行一个文件名后缀为.py的脚本，因为在windows下文件名对文件的打开方式起了决定性作用。</p><p></p><div class="comt-meta wrapper"><span class="comt-author"><a target="_blank" href="javascript:;">Xander663</a><div class="tooltip tooltip1"><p><i class="fa fa-user" aria-hidden="true"></i>&nbsp;&nbsp;&nbsp;Xander663</p><p><i class="fa fa-envelope" aria-hidden="true"></i>&nbsp;&nbsp;xan***1998@163.com</p></div></span>4年前 (2017-12-29)</div></div></li><!-- #comment-## -->
<li class="comment even thread-even depth-1" id="comment-26799"><span class="comt-f">#0</span><div class="comt-avatar wrapper"><i style="font-size:36px;" class="fa fa-user-circle" aria-hidden="true"></i><div class="tooltip"><p><i class="fa fa-user" aria-hidden="true"></i>&nbsp;&nbsp;&nbsp;j88r</p><p><i class="fa fa-envelope" aria-hidden="true"></i>&nbsp;&nbsp;244***88@qq.com</p></div><div id="runoobvote-id-26799" data-commid = "26799" class="upvotejs"><a class="upvote"></a> <span class="count">305</span></div></div><div class="comt-main" id="div-comment-26799"><p>再解释一下第一行代码 <span class="marked">#!/usr/bin/python3</span></p>

<p>这句话仅仅在 linux 或 unix 系统下有作用，在 windows 下无论在代码里加什么都无法直接运行一个文件名后缀为 .py 的脚本，因为在 windows 下文件名对文件的打开方式起了决定性作用。</p>

<p>这个理论不完全正确，至少我知道的不是这样，我在WIN下安装了 64 位的 python,然后下载了 32 位的 embeddable 版，然后在第一行加了这个，把脚本指向 32 位 python 的位置，然后运行正常，是按 32 位版的运行。</p>

<p>至于原因，现在 python 安装的时候会在 windows 目录下放两个文件 py.exe 和 pyw.exe，然后文件类型指向这个这两个文件，可能是由这两个文件判断由哪个 python.exe 去执行脚本。</p>

<div class="comt-meta wrapper"><span class="comt-author"><a target="_blank" href="javascript:;">j88r</a><div class="tooltip tooltip1"><p><i class="fa fa-user" aria-hidden="true"></i>&nbsp;&nbsp;&nbsp;j88r</p><p><i class="fa fa-envelope" aria-hidden="true"></i>&nbsp;&nbsp;244***88@qq.com</p></div></span>4年前 (2018-04-30)</div></div></li><!-- #comment-## -->
<li class="comment odd alt thread-odd thread-alt depth-1" id="comment-36514"><span class="comt-f">#0</span><div class="comt-avatar wrapper"><i style="font-size:36px;" class="fa fa-user-circle" aria-hidden="true"></i><div class="tooltip"><p><i class="fa fa-user" aria-hidden="true"></i>&nbsp;&nbsp;&nbsp;tengjiexx</p><p><i class="fa fa-envelope" aria-hidden="true"></i>&nbsp;&nbsp;104***8544@qq.com</p><p><i class="fa fa-external-link" aria-hidden="true"></i> <a rel="nofollow" target="_blank" href="https://blog.csdn.net/fenglongmiao/article/details/80319875">&nbsp;&nbsp;参考地址</a></p></div><div id="runoobvote-id-36514" data-commid = "36514" class="upvotejs"><a class="upvote"></a> <span class="count">743</span></div></div><div class="comt-main" id="div-comment-36514"><p data-title="#!/usr/bin/python3 和 #!/usr/bin/env python3 的区别" data-commid="38598">脚本语言的第一行，目的就是指出，你想要你的这个文件中的代码用什么可执行程序去运行它，就这么简单。</p>

<p><strong>#!/usr/bin/python3</strong> 是告诉操作系统执行这个脚本的时候，调用 /usr/bin 下的 python3 解释器；</p>

<p><strong>#!/usr/bin/env python3</strong> 这种用法是为了防止操作系统用户没有将 python3 装在默认的 /usr/bin 路径里。当系统看到这一行的时候，首先会到 env 设置里查找 python3 的安装路径，再调用对应路径下的解释器程序完成操作。</p>

<p><strong>#!/usr/bin/python3</strong> 相当于写死了 <strong>python3</strong> 路径;</p>

<p><strong>#!/usr/bin/env python3</strong> 会去环境设置寻找 python3 目录，<strong>推荐这种写法</strong>。</p><div class="comt-meta wrapper"><span class="comt-author"><a target="_blank" href="/note/36514">tengjiexx</a><div class="tooltip tooltip1"><p><i class="fa fa-user" aria-hidden="true"></i>&nbsp;&nbsp;&nbsp;tengjiexx</p><p><i class="fa fa-envelope" aria-hidden="true"></i>&nbsp;&nbsp;104***8544@qq.com</p><p><i class="fa fa-external-link" aria-hidden="true"></i> <a rel="nofollow" target="_blank" href="https://blog.csdn.net/fenglongmiao/article/details/80319875">&nbsp;&nbsp;参考地址</a></p></div></span>3年前 (2018-11-01)</div></div></li><!-- #comment-## -->
	</ol>
	<div class="pagenav">
			</div>
</div>
<div id="respond" class="no_webshot"> 
		<div class="comment-signarea" style="display:none; padding: 20px 20px;"> 
	<h3 class="text-muted" id="share_code" style="color: #799961;"><i class="fa fa-pencil-square-o" aria-hidden="true"></i> 点我分享笔记</h3>
	<!--
	<p style="font-size:14px;">笔记需要是本篇文章的内容扩展！</p><br>
	<p style="font-size:12px;"><a href="//www.runoob.com/tougao" target="_blank">文章投稿，可点击这里</a></p>
	<p style="font-size:14px;"><a href="/w3cnote/runoob-user-test-intro.html#invite" target="_blank">注册邀请码获取方式</a></p>
		<h3 class="text-muted"><i class="fa fa-info-circle" aria-hidden="true"></i> 分享笔记前必须<a href="javascript:;" class="runoob-pop">登录</a>！</h3>
		<p><a href="/w3cnote/runoob-user-test-intro.html#invite" target="_blank">注册邀请码获取方式</a></p>-->
	</div>
		
	<form action="/wp-content/themes/runoob/option/addnote.php" method="post" id="commentform" style="display:none;">
		<div class="comt">
			<div class="comt-title">
				<i style="font-size:36px;" class="fa fa-user-circle" aria-hidden="true"></i>				<p><a id="cancel-comment-reply-link" href="javascript:;">取消</a></p>
			</div>
			<div class="comt-box">
			<div id="mded"></div>
			
				<div class="comt-ctrl">
					<div class="comt-tips"><input type='hidden' name='comment_post_ID' value='7279' id='comment_post_ID' />
<input type='hidden' name='comment_parent' id='comment_parent' value='0' />
</div>
					<button type="submit" name="submit" id="submit" tabindex="5"><i class="fa fa-pencil" aria-hidden="true"></i> 分享笔记</button>
				</div>
			</div>
		
				
					<div class="comt-comterinfo"> 
						<ul id="comment-author-info">
							<li class="form-inline"><label class="hide" for="author">昵称</label><input class="ipt" type="text" name="author" id="author" value="" tabindex="2" placeholder="昵称"><span class="text-muted">昵称 (必填)</span></li>
							<li class="form-inline"><label class="hide" for="email">邮箱</label><input class="ipt" type="text" name="email" id="email" value="" tabindex="3" placeholder="邮箱"><span class="text-muted">邮箱 (必填)</span></li>
							<li class="form-inline"><label class="hide" for="url">引用地址</label><input class="ipt" type="text" name="url" id="url" value="" tabindex="4" placeholder="引用地址"><span class="text-muted">引用地址</span></li>
						</ul>
					</div>
				
			
		</div>

	</form>
	</div>
<script type="text/javascript">
$(function() {
	//初始化编辑器
	
	var editor = new Simditor({
	  textarea: $('#mded'),
	  placeholder: '写笔记...',
	  upload:false,
	 // upload: {url:'/api/comment_upload_file.php',params: null,fileKey: 'upload_file',connectionCount: 1,leaveConfirm: '文件正在上传，您确定离开?'},
	  defaultImage: 'https://www.runoob.com/images/logo.png',
	  codeLanguages: '',
	  autosave: 'editor-content',
	  toolbar: [  'bold','code','ul','ol','image' ]
	});
	editor.on('selectionchanged', function() {
		$(".code-popover").hide();
	});

	// 提交数据
	$("#share_code").click(function() {
		$(".comment-signarea").hide();
		$("#commentform").show();
		
	});
	$("#user_add_note").click(function() {
		$(".comment-signarea").hide();
		$("#commentform").show();
		$('html, body').animate({
       	    scrollTop: $("#respond").offset().top
    	}, 200);
	});

	// 提交笔记
	var commentform=$('#commentform');
	commentform.prepend('<div id="comment-status" style="display:none;" ></div>');
	var statusdiv=$('#comment-status');
	
	commentform.submit(function(e){
		e.preventDefault();
		var noteContent = editor.getValue();
		// console.log(noteContent);
		noteContent = noteContent.replace(/<pre><code>/g,"<pre>");
		noteContent = noteContent.replace(/<\/code><\/pre>/g,"</pre>");
		
		// 系列化表单数据
		var comment_parent = 0;
		var is_user_logged_in = $("#is_user_logged_in").val();
		var comment_post_ID =  7279;
		var _wp_unfiltered_html_comment = $("#_wp_unfiltered_html_comment").val();
		var comment = noteContent;
		var author = $("#author").val();
		var url = $("#url").val();
		var email = $("#email").val();
		if(isBlank(author) && is_user_logged_in==0) {
			statusdiv.html('<p  class="ajax-error">请输入昵称！</p>').show();
		} else if(isBlank(email)  && is_user_logged_in==0) {
			statusdiv.html('<p  class="ajax-error">请输入邮箱！</p>').show();
		} else {
			// var formdata=commentform.serialize() + "&comment=" + noteContent ;
			// 添加状态信息
			statusdiv.html('<p>Processing...</p>').show();
			// 获取表单提交地址
			var formurl=commentform.attr('action');
			
			// 异步提交
			$.ajax({
					type: 'post',
					url: formurl,
					dataType:'json',
					data: {"comment_parent":comment_parent,"comment_post_ID":comment_post_ID, "_wp_unfiltered_html_comment":_wp_unfiltered_html_comment,"comment":comment,"url":url, "email":email,"author":author},
					error: function(XMLHttpRequest, textStatus, errorThrown){
					statusdiv.html('<p class="ajax-error" >数据不完整或表单提交太快了！</p>').show();
				},
				success: function(data, textStatus){
					if(data.errorno=="0") {
						$("#submit").prop('disabled', true);
						statusdiv.html('<p class="ajax-success" >笔记已提交审核，感谢分享笔记！</p>').show();
						alert('笔记已提交审核，感谢分享笔记！');
					}else{
						statusdiv.html('<p class="ajax-error" >'+data.msg+'</p>').show();
					}
					commentform.find('textarea[name=comment]').val('');
				}
			});
			setTimeout(function(){
		        $("#submit").prop('disabled', false);
		    }, 10*1000);
		}
		return false;

	});
	$(".comt-author").click(function() {
		href = $(this).children("a").attr("href");
		if(href.indexOf("/note/")!=-1) {
			var win = window.open(href, '_blank');
  			win.focus();
		}
	});
	$(".comt-meta span").hover(function(){
		$(this).children(".tooltip").css({ "opacity": 1, "pointer-events": "auto"});
	},function(){
		$(this).children(".tooltip").removeAttr("style");
	});
	/*
	$(".wrapper i").hover(function(){
		$(this).siblings(".tooltip").css({ "opacity": 1, "pointer-events": "auto"});
	},function(){
		$(this).siblings(".tooltip").css({ "opacity": 0, "pointer-events": "auto"});
	});
	*/
	//Upvote.create('runoobvote-id', {callback: vote_callback});
	var ajaxurl = 'https://www.runoob.com/wp-admin/admin-ajax.php';
	var callback = function(data) {
		//console.log($('#runoobvote-id').upvote('upvoted'));
		//console.log($('#runoobvote-id').upvote('downvoted'));
		//console.log(data);
		_vote_action = data.action;
		id_arr = data.id.split('-');
		um_id= id_arr[2];
		//console.log(um_id);
		
		var re = /^[1-9]+/;
		if (re.test(um_id)) { 
			var ajax_data = {
				_vote_action: _vote_action,
				action: "pinglun_zan",
				um_id: um_id,
				um_action: "ding"
			};
			//console.log(ajax_data);
			$.post(ajaxurl,ajax_data,function(status){
				//if(status.vote_num>999) {
				//	_voteHtml = '<span style="display: block; text-align: center;font-size: 20px; color: #6a737c; margin: 8px 0;">'+kFormatter(status.vote_num) +'</span>';
				//	$("#runoobvote-id-" + um_id + " .count").hide().after(_voteHtml);
				//}
				
			});
		}
	};
	if($('#comments').length && $('.upvotejs').length){
		$('.upvotejs').upvote({id: 7279, callback: callback});
	
		$.post(ajaxurl,{"action":"pinglun_zan","postid":7279},function(data){  
			$(data).each(function(key,value) {
				$("#runoobvote-id-" + value.commid + " .upvote").addClass(value.upvotejs_class);
				$("#runoobvote-id-" + value.commid + " .downvote").addClass(value.downvote_class);
				$("#runoobvote-id-" + value.commid + " .count").text(value.upvote_count);
			})
		},'json');
		
	}
	
	
});
function isBlank(str) {
    return (!str || /^\s*$/.test(str));
}
function kFormatter(num) {
	// return num;
    return Math.abs(num) > 999 ? Math.sign(num)*((Math.abs(num)/1000).toFixed(1)) + 'k' : Math.sign(num)*Math.abs(num)
}

</script>

<link rel="stylesheet" href="/wp-content/themes/runoob/assets/css/qa.css?1.44">
<link rel="stylesheet" type="text/css" href="https://cdn.staticfile.org/simditor/2.3.6/styles/simditor.min.css" />
<script type="text/javascript" src="https://static.runoob.com/assets/simditor/2.3.6/scripts/module.js"></script>
<script type="text/javascript" src="//static.runoob.com/assets/simditor/2.3.6/scripts/hotkeys.js"></script>
<script type="text/javascript" src="//static.runoob.com/assets/simditor/2.3.6/scripts/uploader.js"></script>
<script type="text/javascript" src="https://cdn.staticfile.org/simditor/2.3.6/lib/simditor.min.js"></script>
<script type="text/javascript" src="https://static.runoob.com/assets/simditor/2.3.6/scripts/simditor-autosave.js"></script>
		<div class="sidebar-box ">
				<div id="ad-336280" >

		<style>	
.responsive_ad1 { display:none; }
@media(min-width: 800px) { .responsive_ad1 {  display:block;margin:0 auto;} }

</style>
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
<!-- 移动版 自动调整 -->
<ins class="adsbygoogle responsive_ad1"
     style="min-width:400px;max-width:728px;width:100%;height:90px;"
     data-ad-client="ca-pub-5751451760833794"
     data-ad-slot="1691338467"
     data-full-width-responsive="true"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
		</div>
				</div>
		
	</div>
</div>
	

<!-- 右边栏 -->
<div class="fivecol last right-column">
<!--
	<div class="tab tab-light-blue" style="text-align: center;">关注微信</div>
	<div class="sidebar-box">
		<a href="http://m.runoob.com/" target="_blank"> <img src="http://www.runoob.com/wp-content/themes/w3cschool.cc/assets/img/qrcode.jpg" alt="移动版"> </a>
		<div class="qqinfo">
		 <a target="_blank" href="http://jq.qq.com/?_wv=1027&k=dOwwKN" id="qqhref">
		<img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="菜鸟家族" title="菜鸟家族"></a>
		<span>(群号：<span id="qqid">365967760</span>)</span>
		</div>
		
	</div>
	-->
<style>
.sidebar-tree .double-li {
	width:300px;
}
.sidebar-tree .double-li li {
    width: 44%;
    line-height: 1.5em;
    border-bottom: 1px solid #ccc;
    float: left;
    display: inline;
}
</style>

	
		<div class="sidebar-box re-box re-box-large">
		<div class="sidebar-box recommend-here" style="margin: 0 auto;">
			<a href="javascript:void(0);" style="font-size: 16px; color:#64854c;font-weight:bold;"> <i class="fa fa-list" aria-hidden="true"></i> 分类导航</a>
		</div>
	<div class="sidebar-box sidebar-cate">
		
		<div class="sidebar-tree" >
			<ul><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> HTML / CSS</a><ul class="double-li"><li><a title="HTML 教程" href="//www.runoob.com/html/html-tutorial.html">HTML 教程</a></li><li><a title="HTML5 教程" href="//www.runoob.com/html/html5-intro.html">HTML5 教程</a></li><li><a title="CSS 教程" href="//www.runoob.com/css/css-tutorial.html">CSS 教程</a></li><li><a title="CSS3 教程" href="//www.runoob.com/css3/css3-tutorial.html">CSS3 教程</a></li><li><a title="Bootstrap3 教程" href="//www.runoob.com/bootstrap/bootstrap-tutorial.html">Bootstrap3 教程</a></li><li><a title="Bootstrap4 教程" href="//www.runoob.com/bootstrap4/bootstrap4-tutorial.html">Bootstrap4 教程</a></li><li><a title="Bootstrap5 教程" href="//www.runoob.com/bootstrap5/bootstrap5-tutorial.html">Bootstrap5 教程</a></li><li><a title="Font Awesome 教程" href="//www.runoob.com/font-awesome/fontawesome-tutorial.html">Font Awesome 教程</a></li><li><a title="Foundation 教程" href="//www.runoob.com/foundation/foundation-tutorial.html">Foundation 教程</a></li></ul></li><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> JavaScript</a><ul class="double-li"><li><a title="JavaScript 教程" href="//www.runoob.com/js/js-tutorial.html">JavaScript 教程</a></li><li><a title="HTML DOM 教程" href="//www.runoob.com/htmldom/htmldom-tutorial.html">HTML DOM 教程</a></li><li><a title="jQuery 教程" href="//www.runoob.com/jquery/jquery-tutorial.html">jQuery 教程</a></li><li><a title="AngularJS 教程" href="//www.runoob.com/angularjs/angularjs-tutorial.html">AngularJS 教程</a></li><li><a title="AngularJS2 教程" href="//www.runoob.com/angularjs2/angularjs2-tutorial.html">AngularJS2 教程</a></li><li><a title="Vue.js 教程" href="//www.runoob.com/vue2/vue-tutorial.html">Vue.js 教程</a></li><li><a title="Vue3 教程" href="//www.runoob.com/vue3/vue3-tutorial.html">Vue3 教程</a></li><li><a title="React 教程" href="//www.runoob.com/react/react-tutorial.html">React 教程</a></li><li><a title="TypeScript 教程" href="//www.runoob.com/typescript/ts-tutorial.html">TypeScript 教程</a></li><li><a title="jQuery UI 教程" href="//www.runoob.com/jqueryui/jqueryui-tutorial.html">jQuery UI 教程</a></li><li><a title="jQuery EasyUI 教程" href="//www.runoob.com/jeasyui/jqueryeasyui-tutorial.html">jQuery EasyUI 教程</a></li><li><a title="Node.js 教程" href="//www.runoob.com/nodejs/nodejs-tutorial.html">Node.js 教程</a></li><li><a title="AJAX 教程" href="//www.runoob.com/ajax/ajax-tutorial.html">AJAX 教程</a></li><li><a title="JSON 教程" href="//www.runoob.com/json/json-tutorial.html">JSON 教程</a></li><li><a title="Echarts 教程" href="//www.runoob.com/echarts/echarts-tutorial.html">Echarts 教程</a></li><li><a title="Highcharts 教程" href="//www.runoob.com/highcharts/highcharts-tutorial.html">Highcharts 教程</a></li><li><a title="Google 地图 教程" href="//www.runoob.com/googleapi/google-maps-basic.html">Google 地图 教程</a></li></ul></li><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> 服务端</a><ul class="double-li"><li><a title="Python 教程" href="//www.runoob.com/python3/python3-tutorial.html">Python 教程</a></li><li><a title="Python2.x 教程" href="//www.runoob.com/python/python-tutorial.html">Python2.x 教程</a></li><li><a title="Linux 教程" href="//www.runoob.com/linux/linux-tutorial.html">Linux 教程</a></li><li><a title="Docker 教程" href="//www.runoob.com/docker/docker-tutorial.html">Docker 教程</a></li><li><a title="Ruby 教程" href="//www.runoob.com/ruby/ruby-tutorial.html">Ruby 教程</a></li><li><a title="Java 教程" href="//www.runoob.com/java/java-tutorial.html">Java 教程</a></li><li><a title="C 教程" href="//www.runoob.com/c/c-tutorial.html">C 教程</a></li><li><a title="C++ 教程" href="//www.runoob.com/cplusplus/cpp-tutorial.html">C++ 教程</a></li><li><a title="Perl 教程" href="//www.runoob.com/perl/perl-tutorial.html">Perl 教程</a></li><li><a title="Servlet 教程" href="//www.runoob.com/servlet/servlet-tutorial.html">Servlet 教程</a></li><li><a title="JSP 教程" href="//www.runoob.com/jsp/jsp-tutorial.html">JSP 教程</a></li><li><a title="Lua 教程" href="//www.runoob.com/lua/lua-tutorial.html">Lua 教程</a></li><li><a title="Rust 教程" href="//www.runoob.com/rust/rust-tutorial.html">Rust 教程</a></li><li><a title="Scala 教程" href="//www.runoob.com/scala/scala-tutorial.html">Scala 教程</a></li><li><a title="Go 教程" href="//www.runoob.com/go/go-tutorial.html">Go 教程</a></li><li><a title="PHP 教程" href="//www.runoob.com/php/php-tutorial.html">PHP 教程</a></li><li><a title="Django 教程" href="//www.runoob.com/django/django-tutorial.html">Django 教程</a></li><li><a title="Zookeeper 教程" href="//www.runoob.com/w3cnote/zookeeper-tutorial.html">Zookeeper 教程</a></li><li><a title="设计模式" href="//www.runoob.com/design-pattern/design-pattern-tutorial.html">设计模式</a></li><li><a title="正则表达式" href="//www.runoob.com/regexp/regexp-tutorial.html">正则表达式</a></li><li><a title="Maven 教程" href="//www.runoob.com/maven/maven-tutorial.html">Maven 教程</a></li><li><a title="Verilog 教程" href="//www.runoob.com/w3cnote/verilog-tutorial.html">Verilog 教程</a></li><li><a title="ASP 教程" href="//www.runoob.com/asp/asp-tutorial.html">ASP 教程</a></li><li><a title="AppML 教程" href="//www.runoob.com/appml/appml-tutorial.html">AppML 教程</a></li><li><a title="VBScript 教程" href="//www.runoob.com/vbscript/vbscript-tutorial.html">VBScript 教程</a></li></ul></li><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> 数据库</a><ul class="double-li"><li><a title="SQL 教程" href="//www.runoob.com/sql/sql-tutorial.html">SQL 教程</a></li><li><a title="MySQL 教程" href="//www.runoob.com/mysql/mysql-tutorial.html">MySQL 教程</a></li><li><a title="PostgreSQL 教程" href="//www.runoob.com/postgresql/postgresql-tutorial.html">PostgreSQL 教程</a></li><li><a title="SQLite 教程" href="//www.runoob.com/sqlite/sqlite-tutorial.html">SQLite 教程</a></li><li><a title="MongoDB 教程" href="//www.runoob.com/mongodb/mongodb-tutorial.html">MongoDB 教程</a></li><li><a title="Redis 教程" href="//www.runoob.com/redis/redis-tutorial.html">Redis 教程</a></li><li><a title="Memcached 教程" href="//www.runoob.com/Memcached/Memcached-tutorial.html">Memcached 教程</a></li></ul></li><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> 数据分析</a><ul class="double-li"><li><a title="Python 教程" href="//www.runoob.com/python3/python3-tutorial.html">Python 教程</a></li><li><a title="NumPy 教程" href="//www.runoob.com/numpy/numpy-tutorial.html">NumPy 教程</a></li><li><a title="Pandas 教程" href="//www.runoob.com/pandas/pandas-tutorial.html">Pandas 教程</a></li><li><a title="Matplotlib 教程" href="//www.runoob.com/matplotlib/matplotlib-tutorial.html">Matplotlib 教程</a></li><li><a title="Scipy 教程" href="//www.runoob.com/scipy/scipy-tutorial.html">Scipy 教程</a></li><li><a title="R 教程" href="//www.runoob.com/r/r-tutorial.html">R 教程</a></li></ul></li><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> 移动端</a><ul class="double-li"><li><a title="Android 教程" href="//www.runoob.com/w3cnote/android-tutorial-intro.html">Android 教程</a></li><li><a title="Swift 教程" href="//www.runoob.com/swift/swift-tutorial.html">Swift 教程</a></li><li><a title="jQuery Mobile 教程" href="//www.runoob.com/jquerymobile/jquerymobile-tutorial.html">jQuery Mobile 教程</a></li><li><a title="ionic 教程" href="//www.runoob.com/ionic/ionic-tutorial.html">ionic 教程</a></li><li><a title="Kotlin 教程" href="//www.runoob.com/kotlin/kotlin-tutorial.html">Kotlin 教程</a></li></ul></li><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> XML 教程</a><ul class="double-li"><li><a title="XML 教程" href="//www.runoob.com/xml/xml-tutorial.html">XML 教程</a></li><li><a title="DTD 教程" href="//www.runoob.com/dtd/dtd-tutorial.html">DTD 教程</a></li><li><a title="XML DOM 教程" href="//www.runoob.com/dom/dom-tutorial.html">XML DOM 教程</a></li><li><a title="XSLT 教程" href="//www.runoob.com/xsl/xsl-tutorial.html">XSLT 教程</a></li><li><a title="XPath 教程" href="//www.runoob.com/xpath/xpath-tutorial.html">XPath 教程</a></li><li><a title="XQuery 教程" href="//www.runoob.com/xquery/xquery-tutorial.html">XQuery 教程</a></li><li><a title="XLink 教程" href="//www.runoob.com/xlink/xlink-tutorial.html">XLink 教程</a></li><li><a title="XPointer 教程" href="//www.runoob.com/xlink/xlink-tutorial.html">XPointer 教程</a></li><li><a title="XML Schema 教程" href="//www.runoob.com/schema/schema-tutorial.html">XML Schema 教程</a></li><li><a title="XSL-FO 教程" href="//www.runoob.com/xslfo/xslfo-tutorial.html">XSL-FO 教程</a></li><li><a title="SVG 教程" href="//www.runoob.com/svg/svg-tutorial.html">SVG 教程</a></li></ul></li><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> ASP.NET</a><ul class="double-li"><li><a title="ASP.NET 教程" href="//www.runoob.com/aspnet/aspnet-tutorial.html">ASP.NET 教程</a></li><li><a title="C# 教程" href="//www.runoob.com/csharp/csharp-tutorial.html">C# 教程</a></li><li><a title="Web Pages 教程" href="//www.runoob.com/aspnet/webpages-intro.html">Web Pages 教程</a></li><li><a title="Razor 教程" href="//www.runoob.com/aspnet/razor-intro.html">Razor 教程</a></li><li><a title="MVC 教程" href="//www.runoob.com/aspnet/mvc-intro.html">MVC 教程</a></li><li><a title="Web Forms 教程" href="//www.runoob.com/aspnet/aspnet-intro.html">Web Forms 教程</a></li></ul></li><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> Web Service</a><ul class="double-li"><li><a title="Web Service 教程" href="//www.runoob.com/webservices/webservices-tutorial.html">Web Service 教程</a></li><li><a title="WSDL 教程" href="//www.runoob.com/wsdl/wsdl-tutorial.html">WSDL 教程</a></li><li><a title="SOAP 教程" href="//www.runoob.com/soap/soap-tutorial.html">SOAP 教程</a></li><li><a title="RSS 教程" href="//www.runoob.com/rss/rss-tutorial.html">RSS 教程</a></li><li><a title="RDF 教程" href="//www.runoob.com/rdf/rdf-tutorial.html">RDF 教程</a></li></ul></li><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> 开发工具</a><ul class="double-li"><li><a title="Eclipse 教程" href="//www.runoob.com/eclipse/eclipse-tutorial.html">Eclipse 教程</a></li><li><a title="Git 教程" href="//www.runoob.com/git/git-tutorial.html">Git 教程</a></li><li><a title="Svn 教程" href="//www.runoob.com/svn/svn-tutorial.html">Svn 教程</a></li><li><a title="Markdown 教程" href="//www.runoob.com/markdown/md-tutorial.html">Markdown 教程</a></li></ul></li><li style="margin: 0;"><a href="javascript:void(0);" class="tit"> 网站建设</a><ul class="double-li"><li><a title="HTTP 教程" href="//www.runoob.com/http/http-tutorial.html">HTTP 教程</a></li><li><a title="网站建设指南" href="//www.runoob.com/web/web-buildingprimer.html">网站建设指南</a></li><li><a title="浏览器信息" href="//www.runoob.com/browsers/browser-information.html">浏览器信息</a></li><li><a title="网站主机教程" href="//www.runoob.com/hosting/hosting-tutorial.html">网站主机教程</a></li><li><a title="TCP/IP 教程" href="//www.runoob.com/tcpip/tcpip-tutorial.html">TCP/IP 教程</a></li><li><a title="W3C 教程" href="//www.runoob.com/w3c/w3c-tutorial.html">W3C 教程</a></li><li><a title="网站品质" href="//www.runoob.com/quality/quality-tutorial.html">网站品质</a></li></ul></li></ul>			</div>
	
	</div>
	</div>
	<br>
	
	<div class="sidebar-box re-box re-box-large">
		<div class="sidebar-box recommend-here">
			<a href="javascript:void(0);">Advertisement</a>
		</div>
		<div class="re-600160" id="sidebar-right-re">
				<script async src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
		<!-- 侧栏1 -->
		<ins class="adsbygoogle"
		     style="display:inline-block;width:160px;height:600px"
		     data-ad-client="ca-pub-5751451760833794"
		     data-ad-slot="4106274865"></ins>
		<script>
		(adsbygoogle = window.adsbygoogle || []).push({});
		</script>
				</div>
	</div>
</div></div>

</div>

<script>
var aid = 7279;
function coll() {
	$.post( '/wp-content/themes/runoob/option/user/userinfo.php', {aid:aid, action:"collarticle", opt:'add'},function( data ) {
		if(data.error==0) {
			$("#content").find("h1:first").find("a").attr("href","javascript:void(0);");
			$("#content").find("h1:first").find("img").attr("src","http://www.runoob.com/wp-content/themes/runoob/assets/img/coll2.png").css({width:32+"px",height:32+"px"});
		}
		alert(data.msg);
	},'json');
}
</script>


<!-- 反馈对话框开始 -->
<script src="/wp-content/themes/runoob/assets/feedback/stable/2.0/feedback.js?1.0"></script>
<link rel="stylesheet" href="/wp-content/themes/runoob/assets/feedback/stable/2.0/feedback.css?1.0" />
<script type="text/javascript">
$.feedback({
    ajaxURL: '/feedback/runoob_feedback.php',
	html2canvasURL: '/wp-content/themes/runoob/assets/feedback/stable/2.0/html2canvas.js',
	onClose: function () {
         window.location.reload();
    }
});
</script>
<!-- 反馈对话框结束 -->
<button class="feedback-btn feedback-btn-gray">反馈/建议</button>
<!-- 底部 -->
<div id="footer" class="mar-t50">
   <div class="runoob-block">
    <div class="runoob cf">
     <dl>
      <dt>
       在线实例
      </dt>
      <dd>
       &middot;<a target="_blank" href="/html/html-examples.html">HTML 实例</a>
      </dd>
      <dd>
       &middot;<a target="_blank" href="/css/css-examples.html">CSS 实例</a>
      </dd>
      <dd>
       &middot;<a target="_blank" href="/js/js-examples.html">JavaScript 实例</a>
      </dd>
      <dd>
       &middot;<a target="_blank" href="/ajx/ajax-examples.html">Ajax 实例</a>
      </dd>
       <dd>
       &middot;<a target="_blank" href="/jquery/jquery-examples.html">jQuery 实例</a>
      </dd>
      <dd>
       &middot;<a target="_blank" href="/xml/xml-examples.html">XML 实例</a>
      </dd>
      <dd>
       &middot;<a target="_blank" href="/java/java-examples.html">Java 实例</a>
      </dd>
     
     </dl>
     <dl>
      <dt>
      字符集&工具
      </dt>
      <dd>
       &middot; <a target="_blank" href="/charsets/html-charsets.html">HTML 字符集设置</a>
      </dd>
      <dd>
       &middot; <a target="_blank" href="/tags/html-ascii.html">HTML ASCII 字符集</a>
      </dd>
     <dd>
       &middot; <a target="_blank" href="/tags/ref-entities.html">HTML ISO-8859-1</a>
      </dd> 
      <dd>
       &middot; <a target="_blank" href="/tags/html-symbols.html">HTML 实体符号</a>
      </dd>
      <dd>
       &middot; <a target="_blank" href="/tags/html-colorpicker.html">HTML 拾色器</a>
      </dd>
      <dd>
       &middot; <a target="_blank" href="//c.runoob.com/front-end/53">JSON 格式化工具</a>
      </dd>
     </dl>
     <dl>
      <dt>
       最新更新
      </dt>
                   <dd>
       &middot;
      <a href="http://www.runoob.com/python3/python-comprehensions.html" title="Python 推导式">Python 推导式</a>
      </dd>
              <dd>
       &middot;
      <a href="http://www.runoob.com/js/js-class-static.html" title="JavaScript 静态方法">JavaScript 静态...</a>
      </dd>
              <dd>
       &middot;
      <a href="http://www.runoob.com/js/js-class-inheritance.html" title="JavaScript 类继承">JavaScript 类继承</a>
      </dd>
              <dd>
       &middot;
      <a href="http://www.runoob.com/js/js-class-intro.html" title="JavaScript 类(class)">JavaScript 类(c...</a>
      </dd>
              <dd>
       &middot;
      <a href="http://www.runoob.com/python3/python-remove-duplicate-from-list.html" title="Python 移除列表中重复的元素">Python 移除列表...</a>
      </dd>
              <dd>
       &middot;
      <a href="http://www.runoob.com/python3/python3-type-conversion.html" title="Python3 数据类型转换">Python3 数据类...</a>
      </dd>
              <dd>
       &middot;
      <a href="http://www.runoob.com/vue3/vue3-create-project.html" title="Vue3 创建项目">Vue3 创建项目</a>
      </dd>
             </dl>
     <dl>
      <dt>
       站点信息
      </dt>
      <dd>
       &middot;
       <a target="_blank" href="//mail.qq.com/cgi-bin/qm_share?t=qm_mailme&amp;email=ssbDyoOAgfLU3crf09venNHd3w" rel="external nofollow">意见反馈</a>
       </dd>
      <dd>
       &middot;
      <a target="_blank" href="/disclaimer">免责声明</a>
       </dd>
      <dd>
       &middot;
       <a target="_blank" href="/aboutus">关于我们</a>
       </dd>
      <dd>
       &middot;
      <a target="_blank" href="/archives">文章归档</a>
      </dd>
    
     </dl>
    
     <div class="search-share">
      <div class="app-download">
        <div>
         <strong>关注微信</strong>
        </div>
      </div>
      <div class="share">
      <img width="128" height="128" src="/wp-content/themes/runoob/assets/images/qrcode.png" />
       </div>
     </div>
     
    </div>
   </div>
   <div class="w-1000 copyright">
     Copyright &copy; 2013-2022    <strong><a href="//www.runoob.com/" target="_blank">菜鸟教程</a></strong>&nbsp;
    <strong><a href="//www.runoob.com/" target="_blank">runoob.com</a></strong> All Rights Reserved. 备案号：<a target="_blank" rel="nofollow" href="https://beian.miit.gov.cn/">闽ICP备15012807号-1</a>
   </div>
  </div>
  <div class="fixed-btn">
    <a class="go-top" href="javascript:void(0)" title="返回顶部"> <i class="fa fa-angle-up"></i></a>
    <a class="qrcode"  href="javascript:void(0)" title="关注我们"><i class="fa fa-qrcode"></i></a>
    <a class="writer" style="display:none" href="javascript:void(0)"   title="标记/收藏"><i class="fa fa-star" aria-hidden="true"></i></a>
    <!-- qrcode modal -->
    <div id="bottom-qrcode" class="modal panel-modal hide fade in">
      <h4>微信关注</h4>
      <div class="panel-body"><img alt="微信关注" width="128" height="128" src="/wp-content/themes/runoob/assets/images/qrcode.png"></div> 
    </div>
  </div>

 <div style="display:none;">
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-84264393-2"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-84264393-2');
</script>
</div>
<script>
window.jsui={
    www: 'https://www.runoob.com',
    uri: 'https://www.runoob.com/wp-content/themes/runoob'
};
</script>

<script src="https://static.runoob.com/assets/libs/hl/run_prettify.js"></script>
<script src="/wp-content/themes/runoob/assets/js/main.min.js?v=1.131"></script>

</body>
</html>


"""

import re
"""
1. 获取所有的div标签
"""  
# result = re.findall("<div.*?</div>", text,re.DOTALL)    
result = re.findall("<div[\d\D]*?</div>", text)   
print(result)
           

"""
2. 获取包含某个属性的标签
"""  
result = re.findall("""<div\sclass="container logo-search".*?</div> """,text,re.DOTALL)
print(result)

"""
3. 获取所有id = "even"的div标签
"""  
result = re.findall("""<div\sclass="tutintro".*?</div> """,text,re.DOTALL)
print(result)


"""
4. 获取某个标签属性的值【分组（）】
"""  
# 获取div的class 属性的值
result = re.findall(""" <div\sclass="(.*?)".*?</div>""",text,re.DOTALL)
print(result)

# 获取a标签的href属性的值
result = re.findall("""<a.*?href="(.*?)".*?</a>""",text,re.DOTALL)
print(result)

"""
5. 获取div里面所有的职位信息
"""  
result1 = re.findall("<h2>(.*?)</h2>", text,re.DOTALL)
print(result1)


"""
新浪热搜榜爬虫
 
  
"""
import requests
import re

# 1.定义要爬取的url
url = "https://s.weibo.com/weibo?q=%23%E5%BC%A0%E8%89%BA%E8%B0%8B%E5%A4%AA%E6%87%82%E4%B8%AD%E5%9B%BD%E4%BA%BA%E7%9A%84%E6%B5%AA%E6%BC%AB%E4%BA%86%23&Refer=top"
# 定制请求头
headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55"}
# 发送请求得到相应
response = requests.get(url,headers = headers)
# 解码相应content ：html字符串
content = response.content.decode("utf-8")

print(content)

#2.使用正则表达式提取数据

contents = re.findall("""<td\sclass="td-02">.*?<a.*?>(.*?)</a>""", content,re.DOTALL)[1:]
print(contents)

hots = re.findall("""<td\sclass="td-02">.*?<span>(.*?)</span>""", content,re.DOTALL)

resous = []
for ct,hot in zip(contents,hots):
    resou = {"content":ct,"hot":hot}
    resous.append(resou)
print(resous)  


"""
爬虫  下载图片
 
  
"""
# 下载一张图片
import requests

# 设置请求头
headers = {
    "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55"}
url = "https://tse1-mm.cn.bing.net/th/id/OIP-C.4fMJTZ2mTRrE8X-i4mBY4wHaLH?pid=ImgDet&rs=1"
# 发送get 请求
response = requests.get(url,headers = headers)
# 变成为 bytes流数据
content = response.content
print(content)

# 保存图片
with open("girl.jpeg","wb") as f:
    f.write(content)
 



# 爬取 下载多张图片
import requests
import re
pic = input("你想爬取什么图片？\n")

# 被爬取的url
url = "https://image.baidu.com/search/index?ct=201326592&z=&tn=baiduimage&ipn=r&w&istype=2&ie=utf-8&oe=utf-8&cl=2&lm=-1&st=-1&fr=&fmq=1644043267536_R&ic=&se=&sme=&width=&height=&face=0&hd=0&latest=1&copyright=0"
# 设置请求头
headers = {
     "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55","accept":"image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"}   
kw = {"word":pic}
# 发送get 请求
response = requests.get(url,headers = headers,params = kw)
  

# 对response 对象解码
content = response.content.decode("utf-8")

# 数据提取
detail_urls = re.findall("""<a.*?href="(.*?)".*?</a>""", content,re.DOTALL)
print(detail_urls)

# 图片的下载
i = 0
for detail_url in detail_urls:
    try:
        # 得到狗的图片响应
        response = requests.get(detail_url,headers =headers)
        # 得到狗的图片的内容（bytes流数据）
        content = response.content
        if detail_url[-3:] == "jpg":
            with open("{}.jpg".format(i),"wb") as f:
                f.write(content)
        elif detail_url[-3:]=="jpeg":
            with open("{}.jpeg".format(i),"wb") as f:
                f.write(content)
        elif detail_url[-3:]=="png":
            with open("{}.png".format(i),"wb") as f:
                f.write(content)
        elif detail_url[-3:]=="bmp":
            with open("{}.bmp".format(i),"wb") as f:
                f.write(content)
        else:
            continue
    except:
        continue
    i +=1
    

    





