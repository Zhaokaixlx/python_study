# import time
# # s_time = time.time()
# # time.sleep(3)
# # print(f"cost{time.time()-s_time}")
# print(time.localtime())
# print(time.gmtime())
# print(time.mktime(time.localtime()))
# print(time.strftime("%Y--%m--%d  %H:%M:%S",time.localtime()))
# time_str = time.strftime("%Y--%m--%d  %H:%M:%S",time.localtime())
# print(time.strptime(time_str,"%Y--%m--%d  %H:%M:%S"))


# datetime 模块
import datetime
d = datetime.datetime.now()
print(d)
print(d.timetuple())
print(d+datetime.timedelta(7,hours=5))
print(d.replace(year=2222,month=1,day=1))

