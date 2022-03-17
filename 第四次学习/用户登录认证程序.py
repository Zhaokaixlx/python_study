# 1.确定用户的输入信息的结构
# 2.把账户信息读到内存，为了方便使用，可以改为list、dict
accounts = {   #"zhao kai":["zhao kai","zk6023080","1"]
     }
f= open("account.db","r")
for line in f:
    line = line.strip().split(",")
    accounts[line[0]] = line
print(accounts)
# 3.搞个loop，要求用户输入账号信息，去判断
while True:
    user = input("username:").strip()
    if user not in accounts:
          print("该用户未注册......")
          continue
    elif accounts[user][2]=="1":
         print("此账户已经锁定，请联系管理员..")
         continue
    count = 0
    while count < 3:
        password = input("your passwords:").strip()
        if password == accounts[user][1]:
            print(f"welcome {user}....登陆成功.....")
            exit("bye......")
        else:
            print("wrong password.........")
    count +=1
    if count == 3:
        print(f"你输错了{count}次密码，需要锁定账户{user}，抱歉........")
        # 1.先改在内存中dict账户信息的 用户状态
        # 2.把dict里面的数据转成原account.db数据格式，并且  存回文件
        accounts[user][2]="1"
        f2 = open("account.db", "w")
        for user,val in accounts.items():
            line = ",".join(val) + "\n"  # 把列表再转化为字符
            f2.write(line)
        f2.close()
        exit("bye...........")
