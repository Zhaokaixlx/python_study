# import  random
# import  string
# count = 0
# while count < 3:
#     car_num = [] # 存储用户输入的车牌号
#     for i in range(20):
#         n1 = random.choice(string.ascii_uppercase)
#         n2 = "".join(random.sample(string.ascii_uppercase+string.digits,5))
#         c_num=f"晋{n1}-{n2}"
#         car_num .append(c_num) # 把生成的号码添加到列表
#         print(i+1,c_num)
#     choice = input("输入你喜欢的号:").strip()
#     if choice in car_num:
#         print(f"恭喜你选择了新的车牌号码为：{choice}")
#         exit("祝您开车一路顺风!!!")
#     else:
#         print("不合法的选择.............")
#     count += 1
import random
import string
count = 0
while count < 3:
    car_num = []
    for i in range(20):
        n1 = random.choice(string.ascii_uppercase)
        n2 ="".join(random.sample(string.ascii_uppercase+string.digits,5))
        c_num = f"沪{n1}-{n2}"
        car_num.append(c_num)
        print(i+1,c_num)
    choice = input("请选择你的车牌号：").strip()
    if choice in car_num:
        print(f"恭喜你选择了车牌号，祝您开车一路顺风,车牌号为:{choice}")
        exit("祝您好运...........")
    else:
        print("您的选择不合法..........")
    count += 1


