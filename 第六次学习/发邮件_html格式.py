 # -*- coding:utf-8 -*-
# created by zhao kai

import smtplib
from email.mime.text import MIMEText # 邮件正文
from email.header import Header  # 邮件头

# 登录

smtp_obj = smtplib.SMTP_SSL("smtp.qq.com", 465)
smtp_obj.login("1365980632@qq.com ", "irlvixpmqkcghfgd")

# 设置邮件内容

mail_body = '''
    <h5>你好, 我叫赵凯，找您有点儿事儿</h5>
    <p>
        xxxxxxx.. <a href="http://wx1.sinaimg.cn/mw1024/5ff6135fgy1gdnghz2vbsg205k09ob2d.gif"> 这是我的照片</a></p>
    </p>
'''

msg = MIMEText(mail_body, "html", "utf-8")

msg["From"] = Header("来自凯同学的问候","utf-8") # 发送者
msg["To"] = Header("xxx","utf-8") # 接收者
msg["Subject"] = Header("赵凯的信","utf-8") # 主题

# 发邮件

smtp_obj.sendmail("1365980632@qq.com", ["xx","xx@qq.com"],msg.as_string())
