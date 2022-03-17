def register_api():
    stu_data = {}
    print("欢迎来到清华大学".center(50,"+"))
    print("请完成学籍注册")
    name = input("name:").strip()
    age = input("age:").strip()
    phone = input("phone:").strip()
    id_num = input("identification:").strip()
    course_list = ["python 开发","linux 云计算","网络安全","数据科学与分析","人工智能"]
    for index,course in enumerate(course_list):
        print(f"{index+1}.{course}")
    selected_course = input("选择想选的课程：")
    if selected_course.isdigit():
        selected_course = int(selected_course)
        if selected_course>=0 and selected_course<len(course_list):
           picked_course = course_list[selected_course]
        else:
           exit("不合法的选项....")
    else:
        exit("非法输入..........")
    stu_data["name"] = name
    stu_data["age"] = age
    stu_data["phone"] = phone
    stu_data["identification"] = id_num
    stu_data["course"] = picked_course
    return stu_data
# def commit_to_db(filename,stu_data):
# student_data = register_api()
# print(student_data)
#     f = open(filename,"a")
#     row = f"{stu_data['name']},{stu_data['age']},{stu_data['phone']},{stu_data['id_num']},{stu_data['course']}\n"
#     f.write(row)
#     f.close()
