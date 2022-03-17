# def sayhi(name,age):
#     print(f"hello,my name is {name},i am {age} old ...")
# sayhi("zhaokai",25)

# 形参  实参
# def cal(x,y):
#     res = x**y
#     print(res)
# a=5
# b=4
# cal(a,b)

# 默认参数 cn
# def stu_register(name,age,course,country="cn"):
#     print("------注册学生信息-------")
#     print("姓名：",name)
#     print("年龄：", age)
#     print("课程：", course)
#     print("国家：", country)
# stu_register("zhao kai",25,"python_devops")
# stu_register("zhao li",23,"lingchauangyixue")
# stu_register("zhang xiue",49,"laoniandxue",country="jp")

# 关键参数  位置参数》关键参数 优先级
# def stu_register(name,age,course,country="cn"):
#     print("------注册学生信息-------")
#     print("姓名：",name)
#     print("年龄：", age)
#     print("课程：", course)
#     print("国家：", country)
# stu_register("zhao kai",25,"python_devops")
# stu_register("zhao li",23,"lingchauangyixue")
# stu_register("zhang xiue",country="jp",course="laoniandxue",age=49)

# 非固定参数：函数定义的时候不确定用户想要传入多少个参数，就可以使用非固定参数
# def stu_register(name,age,course,*args,**kwargs):
#     print(name,age,course,args,kwargs)
# # stu_register("zhaokai",25,"python_devope")
# stu_register("zhao li",23,"linchuangyixue","fm","girl",18035876572,addr="山西吕梁",hometown="圪洞镇")

def stu_register(name, age, course='PY', country='CN'):
    print("----注册学生信息------")
    print("姓名:", name)
    print("age:", age)
    print("国籍:", country)
    print("课程:", course)
    if age > 22:
        return False
    else:
        return True,age,name,course
registriation_status = stu_register("王山炮", 33, course="PY全栈开发", country='JP')
print(registriation_status)
if registriation_status:
    print("注册成功")
else:
    print("too old to be a student.")

# 返回执行结果
# 程序执行，一遇到return ,就代表着函数的结束









