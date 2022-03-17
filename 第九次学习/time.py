import time
# 打印当前的时间
# 这个是unix时间
# now = time.time()
# print(now)

"""
计算某个程序运行的时间
"""
# def sum1(times):
#     z=0
#     for i in range(1,times+1):
#         z = z +i
#     print(z)
# start_time = time.time()
# result = sum1(1000000)
# end_time = time.time()
# print("程序运行的时间为{:.2f}s.".format(end_time-start_time))

"""
延时打印
"""
for i in range(3):
    print("sleep!!!!")
    time.sleep(2)

