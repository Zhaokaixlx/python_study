# f = open(r"D:\pycharmcode\第九次学习\1.txt",mode="w",encoding="utf-8")
# f.write(""""寻寻觅觅，冷冷清清，凄凄惨惨戚戚。乍暖还寒时候，最难将息。三杯两盏淡酒，怎敌他、晚来风急！雁过也，正伤心，却是旧时相识。/n
# 满地黄花堆积，憔悴损，如今有谁堪摘？守着窗儿，独自怎生得黑！梧桐更兼细雨，到黄昏、点点滴滴。这次第，怎一个愁字了得！""")
# f.flush()
# f.close()

"""
1.删除原文件中的内容，重新xieru
2.创建没有的文件
"""


# 批量写出文件
def out(num):
        i=1
        for i in range(1,num+1):
# 先建一个空文件夹，再加入
                path = fr"D:\pycharmcode\第九次学习\4\{i}.txt"
                f = open(path,mode="w",encoding="utf-8")
                f.write("""桃之夭夭，灼灼其华。之子于归，宜其室家。\n桃之夭夭，有蕡其实。之子于归，宜其家室。\n桃之夭夭，其叶蓁蓁。之子于归，宜其家人。""")
                f.flush()
                f.close()
out(5)

# mode = "a" 追加数据
# f = open(r"D:\pycharmcode\第九次学习\1.txt",mode="a",encoding="utf-8")
# f.write("\n桃之夭夭，灼灼其华。之子于归，宜其室家。\n桃之夭夭，有蕡其实。之子于归，宜其家室。\n桃之夭夭，其叶蓁蓁。之子于归，宜其家人。")
# f.flush()
# f.close()

# with open  r w a  r+  都可
# with open(r"xx\xx\xx.xxx","a") as f :
#     f.write("静夜思")



