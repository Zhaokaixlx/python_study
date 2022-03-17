import pyautogui    # 第三方模块，需要安装才能使用
import pyperclip   # 第三方模块，需要安装才能使用
import time


def get_msg():
    """想发的消息，每条消息空格分开"""
    contents = "我牛逼吗 我是赵凯 导弹已经安装完毕 马上到达你家  这是用python写的  逗你玩玩  璐璐姐 某人到商店买点钞机，挑了两台最贵的，同时问了一下老板为什么这种型号的贵一些，老板告诉他因为这是全智能语音型的。 \
        "
    return contents.split(" ")


def send(msg):
    # 复制需要发送的内容到粘贴板
    pyperclip.copy(msg)
    # 模拟键盘 ctrl + v 粘贴内容Z
    pyautogui.hotkey('ctrl', 'v')
    # 发送消息
    pyautogui.press('enter')


def send_msg(friend):
    # Ctrl + alt + w 打开微信
    pyautogui.hotkey('ctrl', 'alt', 'w')
    # 搜索好友
    pyautogui.hotkey('ctrl', 'f')
    # 复制好友昵称到粘贴板
    pyperclip.copy(friend)
    # 模拟键盘 ctrl + v 粘贴
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(1)
    # 回车进入好友消息界面
    pyautogui.press('enter')
    # 一条一条发送消息
    for msg in get_msg():
        send(msg)
        # 每条消息间隔 2 秒
        time.sleep(2)


friend_names = ['王璐宁',"文件传输助手"]  # 好友列表 ，给自己的微信好友好信息
for friend_name in friend_names:
        send_msg(friend_name)
print("ok")