# import jieba
# jieba.lcut(s)是最常用的中文分词函数，用于精准模式，即将字符串分割成等量的中文词组，返回结果是列表类型。
# print(jieba.lcut("由于中文文本中的单词不是通过空格或者标点符号分割"))

# jieba.lcut(s, cut_all = True)用于全模式，即将字符串的所有分词可能均列出来，返回结果是列表类型，冗余性最大。
# print(jieba.lcut("由于中文文本中的单词不是通过空格或者标点符号分割",cut_all = True))

# jieba.lcut_for_search(s)返回搜索引擎模式，该模式首先执行精确模式，然后再对其中长词进一步切分获得最终结果。
# print(jieba.lcut_for_search("由于中文文本中的单词不是通过空格或者标点符号分割"))
# jieba.add_word()函数，顾名思义，用来向jieba词库增加新的单词。
# jieba.add_word("潮享")
# print(jieba.lcut('潮享教育',cut_all = True))


import wordcloud
txt='I like python. I am learning python'
wd = wordcloud.WordCloud().generate(txt)
wd.to_file('test.jpg')

# import jieba
# txt = '程序设计语言是计算机能够理解和识别用户操作意图的一种交互体系，它按照特定规则组织计算机指令，使计算机能够自动进行各种运算处理。'
# words = jieba.lcut(txt) # 精确分词
# newtxt = ' '.join(words) # 空格拼接
# wd = wordcloud.WordCloud(font_path="msyh.ttc",width = 500,height = 500).generate(newtxt)
# wd.to_file('词云中文例子图.png') # 保存图片


import wordcloud
# from scipy.misc import imread
from imageio import imread
mask = imread(r"D:\pycharmcode\第十一次学习\2.png")
with open(r"D:\pycharmcode\第十一次学习\1.txt", 'r',encoding="utf-8") as file:
    text = file.read()
    wd = wordcloud.WordCloud(background_color="white", \
                        width=800, \
                        height=600, \
                        max_words=200, \
                        max_font_size=80, \
                        mask = mask, \
                        ).generate(text)
# 保存图片
wd.to_file('2词云.jpg')



# from PyQt5.QtWidgets import QApplication, QWidget,QPushButton   #这里引入了PyQt5.QtWidgets模块，这个模块包含了基本的组件。
# from PyQt5.QtGui import QIcon
# import sys
#
# app = QApplication(sys.argv)  #sys.argv是一组命令行参数的列表
# w = QWidget()  #QWidge控件是一个用户界面的基本控件，这里是一个窗口（window）。
# w.resize(550, 250)   #窗口宽250px，高150px。
# w.move(900, 300)  #修改控件位置的的方法
# w.setWindowTitle('潮享教育')   #窗口添加了一个标题
# w.setWindowIcon(QIcon('python.png'))
# w.show()  #show()能让控件在桌面上显示出来
# sys.exit(app.exec_())



####  把python 代码形成一个可以在其他没有安装python的电脑上也可以运行的执行文件
####  在terimnal  上运行  pyinstallder库



