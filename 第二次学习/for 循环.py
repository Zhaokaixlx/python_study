# for i in range(5,10):
#     print(i)
# 猜年龄的游戏代码
# black_girl_age=26
# for i in range(3):
#     guess=int(input("输入你的猜测："))
#     if guess>black_girl_age: # 猜大了
#         print("你讨厌，人家哪有这么老啊......")
#     elif guess<black_girl_age:
#         print("猜小了，人家已经很成熟了呢")
#     else:
#         exit("恭喜你，你猜对了啊") #退出程序
# 打印从0-100的 奇偶数
# for i in range(101):
#     if i % 2==0: # 代表是偶数
#         print(f"{i}是偶数")
#     else:
#         print(f"{i}是奇数")
# 打印各楼层的房间号  for 循环的嵌套
# for i in range(1,6):
#     print(f"------------{i}层------------")
#     for j in range(1,10):
#         print(f"{i}-{i}0{j}")
# 打印楼层小程序，需求改变，遇到第三层时，不打印房间号，其它层打印
# 用break and continue  函数
# for i in range(1,6):
#     print(f"------------{i}层------------")
#     if i == 3:
#         print("三层不走.....")
#         continue  # 跳过本次循环，进入下一次
#     for j in range(1,10):
#         print(f"{i}-{i}0{j}")
# 遇到404房间，遇到鬼屋，扑街了
# for i in range(1,6):
#     print(f"------------{i}层------------")
#     for j in range(1,10):
#         if i == 4 and j == 4:
#            print("遇到鬼屋，扑街了.........")
#            break  # 跳过大循环
#         print(f"{i}-{i}0{j}")
# 打印三角形
# for i in range(10):
#     if i<=5:
#         print("*"*i)
#     else:
#         print((10-i)*"*")









