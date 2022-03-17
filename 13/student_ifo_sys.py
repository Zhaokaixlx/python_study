# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:09:59 2022

@author: zhaokai
"""

"""
学生信息管理系统的开发
1.需求分析
2.系统设计
3.系统开发必备
4.主函数设计
5.学生信息维护模块设计
6.查询、统计模块
7.排序模块设计
8.项目打包

"""
import os
filename = "student.txt"
def main():
    while True:
        muen()
        choice = int(input("请选择："))
        if choice in [0,1,2,3,4,5,6,7]:
            if choice == 0:
                answer = input("您确定要推出系统吗？y\n")
                if answer =="y" or answer =="Y":
                    print("谢谢您的使用！！！")
                    break
                else :
                    continue
            elif choice ==1:
                insert()
            elif choice ==2:
                search()
            elif choice ==3:
                delete()
            elif choice ==4:
                modify()
            elif choice ==5:
                sort()
            elif choice ==6:
                total()
            elif choice ==7:
                show()



def muen():
    print("-----------学生信息管理系统-------------------")
    print("------------功能菜单---------------------------------")
    print("\t\t\t\t\t\t\t1.录入学生信息")
    print("\t\t\t\t\t\t\t2.查找学生信息")
    print("\t\t\t\t\t\t\t3.删除学生信息")
    print("\t\t\t\t\t\t\t4.修改学生信息")
    print("\t\t\t\t\t\t\t5.排序")
    print("\t\t\t\t\t\t\t6.统计学生总人数")
    print("\t\t\t\t\t\t\t7.显示学生信息")
    print("\t\t\t\t\t\t\t0.退出")
    print("++++++++++++++++++++++++++++++++++++++++++++++")
def insert():
    student_list = []
    while True:
       id = input("请输入id（例如：1001）")
       if not id:
           break
       name = input("请输入姓名：")
       if not name:
           break
       try:
           englist = int(input("请输入英语成绩："))
           python = int(input("请输入python成绩："))
           java = int(input("请输入java成绩："))
       except:
           print("输入无效，不是有效的成绩")
           continue
       student = {"id":id,"name":name,"englist":englist,"python":python,"java":java}
       student_list.append(student)
       answer = input("是否继续添加 y\n")
       if answer == "y":
           continue
       else:
           break
    save(student_list)
    print("学生信息录入完毕")
def save(lst):
    try:
        stu_txt = open(filename,"a",encoding="utf-8")
    except:
        stu_txt = open(filename,"w",encoding="utf-8")
    for item in lst:
        stu_txt.write(str(item)+"\n")
    stu_txt.close()

def delete():
    while True:
        student_id=input('请输入要删除的学生的ID：')
        if student_id!='':
            if os.path.exists(filename):
                with open(filename,'r',encoding='utf-8')as file:
                    student_old=file.readlines()
            else:
                student_old=[]
            flag=False   #标记是否删除
            if student_old:
                with open(filename,'w',encoding='utf-8') as wfile:
                    d={}
                    for item in student_old:
                        d=dict(eval(item)) #将字符串转成字典
                        if d['id']!=student_id:
                            wfile.write(str(d)+'\n')
                        else:
                            flag=True
                    if flag:
                        print(f'id为{student_id}的学生信息已被删除')
                    else:
                        print(f'没有找到ID为{student_id}的学生信息')
            else:
                print('无学生信息')
                break
            show()     #删除之后要重新显示所有学生信息
            answer=input('是否继续删除?y/n\n')
            if answer=='y':
                continue
            else:
                break


def search():
    student_query=[]
    while True:
        id=''
        name=''
        if os.path.exists(filename):
            mode=input('按ID查找请输入1，按姓名查找请输入2：')
            if mode=='1':
                id=input('请输入学生ID')
            elif mode=='2':
                name=input('请输入学生姓名:')
            else:
                print('您的输入有误，请重新输入')
                search()
            with open(filename,'r',encoding='utf-8') as rfile:
                student=rfile.readlines()
                for item in student:
                    d=dict(eval(item))
                    if id!='':
                        if d['id']==id:
                           student_query.append(d)
                    elif name!='':
                        if d['name']==name:
                            student_query.append(d)
            #显示查询结果
            show_student(student_query)
            #清空列表
            student_query.clear()
            answer=input('是否要继续查询?y/n\n')
            if answer=='y':
                continue
            else:
                break
        else:
            print('暂未保存学生信息')
            return
def show_student(lst):
    if len(lst)==0:
        print("没有查询到学生信息，无数据显示！！！")
        return
    # 定义标题显示格式
    format_title = "{:*6}\t{:*12}\t{:*8}\t{:10}\t{:*10}\t{:*8}"
    print(format_title.format("id", "姓名","英语成绩","python成绩","java成绩","总成绩"))
    # 定义内容显示格式
    format_data = "{:*6}\t{:*12}\t{:*8}\t{:8}\t{:*10}\t{:*8}"
    for item in lst:
        print(format_data.format(item.get("id"),
                                 item.get("name"),
                                 item.get("english"),
                                 item.get("python"),
                                 item.get("java"),
                                 int(item.get("english"))+int(item.get("python"))+int(item.get("java"))))
    else:
        pass

def modify():
    show()
    if os.path.exists(filename):
        with open(filename,"r",encoding="utf-8") as rfile:
            student_old = rfile.readlines()
    else:
        return
    student_id = input("请输入要修改的学院的id：")
    with open(filename,"w",encoding="utf-8") as wfile:
        for item in student_old:
            d = dict(eval(item))
            if d["id"]==student_id:
                print("找到学生信息，可以修改他的相关信息了！")
                while True:
                    try:
                        d["name"] = input("请输入姓名：")
                        d["english"] = input("请输入英语成绩：")
                        d["python"] = input("请输入python 成绩：")
                        d["java"]  = input("请输入java 的成绩：")
                    except:
                        print("您输入的信息有误，请重新输入！！！")
                    else:
                        break
                wfile.write(str(d)+"\n")
                print("修改成功")
            else:
                wfile.write(str(d)+"\n")
        answer = input("是否要继续修改其他学生的信息呢？y\n\n")
        if answer == "y":
            modify()

def sort():
   show()
   if os.path.exists(filename):
       with open(filename,'r',encoding='utf-8') as rfile:
           student_list=rfile.readlines()
       student_new=[]
       for item in student_list:
           d=dict(eval(item))
           student_new.append(d)
   else:
       return
   asc_or_desc=input('请选择(0.升序 1.降序):')
   if asc_or_desc=='0':
       asc_or_desc_bool=False
   elif asc_or_desc=='1':
       asc_or_desc_bool=True
   else:
       print('您的输入有误，请重新输入')
       sort()
   mode=input('请选择排序方式(1.按英语成绩排序 2.按Python成绩排序 3.按Java成绩排序 0.按总成绩排序):')
   if mode=='1':
       student_new.sort(key=lambda x :int(x['english']),reverse=asc_or_desc_bool)
   elif mode=='2':
       student_new.sort(key=lambda x: int(x['python']), reverse=asc_or_desc_bool)
   elif mode=='3':
       student_new.sort(key=lambda x: int(x['java']), reverse=asc_or_desc_bool)
   elif mode=='0':
       student_new.sort(key=lambda x: int(x['english'])+int(x['python'])+ int(x['java']), reverse=asc_or_desc_bool)
   else:
       print('您的输入有误，请重新输入!!!')
       sort()
   show_student(student_new)

def total():
    if os.path.exists(filename):
        with open(filename,"r",encoding="utf-8") as rfile:
            students = rfile.readlines()
            if students:
                print(f"一共有{len(students)}名学生")
            else:
                print("还没有录入学生信息")

def show():
    student_lst=[]
    if os.path.exists(filename):
        with open(filename,'r',encoding='utf-8') as rfile:
            students=rfile.readlines()
            for item in students:
                student_lst.append(eval(item))
            if student_lst:
                show_student(student_lst)
    else:
        print('暂未保存过数据!!!!')

if __name__ =="__main__":
    main()
