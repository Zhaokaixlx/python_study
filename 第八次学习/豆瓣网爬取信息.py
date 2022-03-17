import re
import requests # 下载网页
import bs4  # 解析网页
headers = {
    "Accept":"text/html,appplication/xhtml+xml.application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language":"zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6"
    "Cache-Control":"private"
    "Connection":"Keep-Alive"
    "Host":"www.douban.com",
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55"
}
page_obj = requests.get("https://www.douban.com/group/topic/187973616/?_dtcc=1&_i=2942240OW06uOB",headers=headers)
print(page_obj.text)
bs4_obj = bs4.BeautifulSoup(page_obj.text,"lxml")

# fetch emails
mail_list = []
comment_eles = bs4_obj.find_all("div",attrs={"class":"reply-doc"})
for ele in comment_eles:
    comment_ele = ele.find("p",attrs={"calss":"reply-content"})
    email_addr = re.search("\w+@\w.\w+",comment_ele.text,flags=re.A)
    if email_addr:
        pub_time = ele.find("span",attrs={"class":"pubtime"})
        mail_list.append([email_addr.group(),pub_time.text])
    print(mail_list)




