# -*- coding:utf-8 -*-
# created by zhao kai

import smtplib
from email.mime.text import MIMEText # 邮件正文
from email.header import Header

# 登录

smtp_obj = smtplib.SMTP_SSL("smtp.qq.com", 465)
smtp_obj.login("1365980632@qq.com ", "irlvixpmqkcghfgd")

# 设置邮件内容

msg = MIMEText("你好, 我叫赵凯，找您有点儿事儿", "plain", "utf-8")
msg["From"] = Header("来自凯同学的问候","utf-8") # 发送者
msg["To"] = Header("xxx","utf-8") # 接收者
msg["Subject"] = Header("赵凯的信","utf-8") # 主题

# 发邮件
smtp_obj.sendmail("1365980632@qq.com", ["xxxxxx@qq.com","xxxxxx@qq.com"], msg.as_string())
