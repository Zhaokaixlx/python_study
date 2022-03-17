# -*- coding:utf-8 -*-
# created by zhao kai

import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

# 登录邮件服务器
smtp_obj = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件⼈邮箱中的SMTP服务器，端⼝是25
smtp_obj.login("1365980632@qq.com", "irlvixpmqkcghfgd")  # 括号中对应的是发件⼈邮箱账号、邮箱密码
smtp_obj.set_debuglevel(1)  # 显示调试信息
# 设置邮件头信息
mail_body = '''
 <h5>你好, 我叫赵凯，找您有点儿事儿</h5>
 <p>
 xxxxxxxxxxxxx........

 <p><img src="cid:image1"></p>
 </p>
'''
msg_root = MIMEMultipart('related')  # 允许添加附件、图⽚等
msg_root["From"] = Header("来自凯同学的问候", "utf-8")  # 发送者
msg_root["To"] = Header("xxx", "utf-8")  # 接收者
msg_root["Subject"] = Header("赵凯的信", "utf-8")  # 主题
# 允许添加图⽚
msgAlternative = MIMEMultipart('alternative')
msgAlternative.attach(MIMEText(mail_body, 'html', 'utf-8'))
msg_root.attach(msgAlternative)  # 把邮件正⽂内容添加到msg_root⾥
# 加载图⽚，
fp = open('girl.jpg', 'rb')
msgImage = MIMEImage(fp.read())
fp.close()
# 定义图⽚ ID，在 HTML ⽂本中引⽤
msgImage.add_header('Content-ID', '<image1>')
msg_root.attach(msgImage)  # 添加图⽚到msg_root对象⾥
# 发送
smtp_obj.sendmail("1365980632@qq.com", ["xxxxxx.com",
                                         "xxxxxxx@qq.com"], msg_root.as_string())