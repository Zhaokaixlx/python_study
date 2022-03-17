# -*- coding:utf-8 -*-

import sys
print(sys.argv)
old_str = sys.argv[1]
new_str = sys.argv[2]
filename = sys.argv[3]
#  1.load into ram
f = open(filename,"r+")
data = f.read()
# 2.count and replace
old_str_count = data.count(old_str)
new_data = data.replace(old_str,new_str)
# 3. clear old filename
f.seek(0)
f.truncate()
# 4.save new data into file
f.write(new_data)
print(f"成功替换字符{old_str}to{new_str}，共{old_str_count}")