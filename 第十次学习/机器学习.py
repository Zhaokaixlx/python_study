# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:23:21 2022

@author: Administrator
"""

"""
1. 机器学习是什么?
从谚语出发--朝霞不出门，晚霞行千里
数据---归纳总结
有用的模型
模型预测 
本质--从数据中挖掘出有用的信息
2. 为什么学习机器学习？
信息爆炸时代--数据量太大，人工已经无法处理
重复性的工作
潜在信息的关联
机器学习有效
应用领域广泛
3.机器学习的应用
垃圾邮件识别
金融量化--- 预测股票 
反欺诈
图像识别 -人脸识别、动物识别、植物识别
自然语言处理
推荐系统
无人驾驶
4.如何让学习机器学习
python
认真学习理论--数学理论
大学数学的基础

"""

"""
识别狗狗
if 
传统模型：单一  特征人为指定  浪费时间取找特征  图像法师遮盖无法判断
改进  
数据整理 -机器重复处理-自动选择特征，形成模型-预测

人：  有特征 找结果
机器： 从结果  找特征  形成模型  预测

"""


# 机器学习环境的搭建[Anaconda]
"""
1.numpy 1.20.3  矩阵运算
2.scipy  1.7.1  数值运算等
3.matplotlib    3.4.3  绘图库
4.pandas  1.3.4  数据清洗【对数据进行的一些操作】、数据读取
5.sklearn  0.24.2  算法库--核心

"""


# 算法
"""
线性回归：利用回归分析，来确定两种或者两种以上
   变量之间的相互依赖的定量关系的一种统计方法。
   1.相关关系
   2.因果关系 --才可以进行线性回归
   3.平行关系
   
"""

"""
极大似然估计：极度自恋，相信自己就是天选之子，自己看到的，就是冥冥之中
最接近真相的 
   
"""



####  sklearn 库  一元线性回归
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
    
"""
相关关系：包含因果关系和平行关系
因果关系：回归分析【原因引起结果，需要明确自变量和因变量】
平行关系：相关分析【无因果关系，不区分自变量和因变量】
"""
data = pd.read_csv("C:/Users/Administrator/Desktop/data.csv")

#绘制散点图，求x和y的相关系数
plt.scatter(data.广告投入, data.销售额)
data.corr()

#估计模型参数，建立回归模型
lr = LinearRegression()

x = data[['广告投入']]
y = data[['销售额']]

#训练模型
lr.fit(x, y)

#第四步、对回归模型进行检验
"""
此处的score指R方
"""
lr.score(x, y)

#第五步、利用回归模型进行预测
lr.predict([[40], [45], [50]])

#查看截距
a = round(lr.intercept_[0], 2)
#查看斜率
b = round(lr.coef_[0][0], 2)

print("线性回归模型为：y = {}x + {}.".format(b, a))





####  sklearn 库  多元线性回归

import pandas as pd
from sklearn.linear_model import LinearRegression

#导入数据
data = pd.read_csv("C:/Users/Administrator/Desktop/多元线性回归.csv",
                   encoding='gbk', engine='python')
print(data)

#打印相关系数矩阵
data[["体重", "年龄", "血压收缩"]].corr()

#第二步，估计模型参数，建立回归模型
"""
建立线性回归模型
x：自变量   y:因变量
"""
lr_model = LinearRegression()
x = data[['体重', '年龄']]
y = data[['血压收缩']]
#训练模型
lr_model.fit(x,y)

#第四步，对回归模型进行检验
"""
score:调整R方，判断自变量对因变量的解释程度
R方越接近于1，自变量对因变量的解释就越好
F检验：方程整体显著性检验
T检验：方程系数显著性检验
score给的是R方.
"""
lr_model.score(x,y)

#第五步，利用回归模型进行预测
lr_model.predict([[80, 60]])

lr_model.predict([[70, 30],[70, 20]])

#查看参数
"""
a:自变量系数
b:截距
"""
a = lr_model.coef_
b = lr_model.intercept_
print("线性回归模型为：y = {:.2f}x1 + {:.2f}x2 + {:.2f}."
      .format(a[0][0], a[0][1], b[0]))
#线性回归模型为：y = 2.14x1 + 0.40x2 + -62.96.






"""
statsmodels   具有很多统计模型的python库
  能够完成很多统计测试、数据探索以及可视化。也包含一些
  经典的统计方法，比如贝叶斯方法
  


"""
#### statsmodels  实现一元线性回归
import pandas as pd

#读取数据
data = pd.read_csv("./data.csv")

#自变量与因变量分离
x = data[['广告投入']]
y = data[['销售额']]

import statsmodels.api as sm

#添加常数项
X = sm.add_constant(x)

#最小二乘法
model = sm.OLS(y, X)
result = model.fit()

#系数
result.params
#y = 3.737x -36.361

#汇总结果
result.summary()

#预测结果
y_pr = result.fittedvalues

#绘制图像
"""
plt.subplots:绘制子图的方式,返回值为fig和ax
fig为将要绘制图像的大小，ax为将要绘制图像的信息
"""
from matplotlib import pyplot as plt

fig,ax = plt.subplots(figsize=(8,6))
#plt.xlim(0,100)
#plt.ylim(0,100)
ax.plot(x, y, 'o', label='data')
ax.plot(x, y_pr, 'r--', label='OLS')
ax.legend(loc='best')




#### statsmodels  实现多元线性回归
import pandas as pd

#读取数据
data = pd.read_csv("./多元线性回归.csv",
encoding='gbk', engine='python')

#自变量与因变量分离
x = data[['体重', '年龄']]
y = data[['血压收缩']]

import statsmodels.api as sm

#添加常数项
X = sm.add_constant(x)

#最小二乘法
model = sm.OLS(y, X)
result = model.fit()

#系数
result.params
"""
y = 2.13体重 + 0.40年龄 -62.96
"""

#汇总结果
result.summary()

"""
通过调整alpha来调整置信水平,默认95%
常用的置信水平为0.05或者是0.01
"""
result.summary(alpha = 0.10)

#预测结果
y_pr = result.fittedvalues





#####  实际案例

import pandas as pd
import statsmodels.api as sm


data = pd.read_excel("./案例.xlsx")

"""
第一步：进行相关性分析,消除变量之间的共线性
"""
r = data[['不良贷款','各项贷款余额','本年累计应收贷款',
'贷款项目个数','本年固定资产投资额']].corr()

"""
第二步：实现多元线性回归
"""
x = data[['各项贷款余额','本年累计应收贷款',
'贷款项目个数','本年固定资产投资额']]
y = data[['不良贷款']]

#添加常数项
X = sm.add_constant(x) 

#最小二乘法
model = sm.OLS(y, X)
result = model.fit()

#查看汇总结果
result.summary(alpha=0.05)


"""
第三步：优化多元线性回归
解决共线性的办法:尽量增大样本容量
"""
x = data[['各项贷款余额','本年累计应收贷款','本年固定资产投资额']]
y = data[['不良贷款']]

#添加常数项
X = sm.add_constant(x) 

#最小二乘法
model = sm.OLS(y, X)
result = model.fit()

#查看汇总结果
result.summary(alpha=0.05)


"""
第四步：进行预测
"""
test = data[:5]
x = test[['各项贷款余额','本年累计应收贷款','本年固定资产投资额']]
X = sm.add_constant(x) 

y_hat = result.predict(X)
print(y_hat)





"""
梯度下降  
  损失函数:相当于下山 ，安装一个GPS
           找到一个最低最低的山谷-最小值点

  1.批量梯度下降BGD
  2.随机梯度下降SGD ：每次一个样本，迭代速度快
  3.小批量梯度下降MBGD：每次更新选择一小部分数据来算
     常用
  学习率的选取：非常重要的参数  建议取小一些 0.01
  Batchsize的大小：
        32、64、168都可以---根据电脑配置选择

"""




"""
逻辑回归
  回归：预测的值为连续值
  分类：预测值为分类变量
概念：用于处理因变量为分类变量的回归问题，常见2分
     也可以处理多种分类问题，它实际上属于一种
     分类方法  
Sigmoid 函数： 自变量：负无穷到正无穷
               值域：[0,1]
本质：将线性回归的结果映射到[0,1]的区间上，完成了
      连续变量到概率的转换，大于0.5类别1，小于0.5
      类别是0。实质上就是完成靓二分任务
评价模型：---混淆矩阵 --
                 TP（ture positive） 5
                 FP(false positive)  4
                 FN(false negative)  2 
                 TN(ture negative)   4
               Accuracy(准确率):
                   acc = (TP+TN)/(TP+TN+FP+FN)
               1     0
          1   5      2
          0   4      4
               Precision(精确率)：
                    p = TP/(TP+FP)
               Recall(召回率):
                   recall = TP/(TP+FN)
               F1-score(F1值)：
                    F1= 2*（ Precision*Recall）/(Precision+Recall)
               这些值都是越大越好！！！！！     
"""
#####   鸢尾花案例
import pandas as pd

#读取数据
"""
train_data:训练集--专门用来训练模型【相当于模拟考试】
test_data:测试集--专门用来测试模型【相当于高考】
"""
train_data = pd.read_excel("./鸢尾花训练数据.xlsx",encoding='utf8')
test_data = pd.read_excel("./鸢尾花测试数据.xlsx",encoding='utf8')

"""
处理训练集数据;
数据重排;变量与标签分离.
"""
train_data.columns
train_X = train_data[['萼片长(cm)', '萼片宽(cm)', '花瓣长(cm)', '花瓣宽(cm)']]
train_y = train_data[['类型_num']]
"""
生成逻辑回归对象,并对训练集数据进行训练
"""
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)

"""
使用训练集数据查看训练效果
"""
#训练集结果
train_predicted = model.predict(train_X)

"""
绘制混淆矩阵;可视化混淆矩阵.
"""
from sklearn import metrics

#绘制混淆矩阵
print(metrics.classification_report(train_y, train_predicted))

#可视化混淆矩阵
def cm_plot(y,yp):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',
                         verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    return plt
        
cm_plot(train_y, train_predicted).show() 

"""
使用测试集数据进行测试
"""
test_X = test_data[['萼片长(cm)', '萼片宽(cm)', '花瓣长(cm)', '花瓣宽(cm)']]
test_y = test_data[['类型_num']]
#预测结果
test_predicted = model.predict(test_X)
#预测概率
test_predicted_pr = model.predict_proba(test_X)

"""
绘制混淆矩阵;可视化混淆矩阵.
"""
from sklearn import metrics

#绘制混淆矩阵
predicted = model.predict(test_X)
print(metrics.classification_report(test_y, test_predicted))

#可视化混淆矩阵
cm_plot(test_y, test_predicted).show() 

"""
进行预测
"""
#导入预测数据
predict_data = pd.read_excel("./鸢尾花预测数据.xlsx",encoding='utf8')

pr_X = predict_data[['萼片长(cm)', '萼片宽(cm)', '花瓣长(cm)', '花瓣宽(cm)']]

#预测结果
predicted = model.predict(pr_X)
#预测概率
predicted_pr = model.predict_proba(pr_X)



#####   贷款风险用户评估
import pandas as pd
import numpy as np
import time

#可视化混淆矩阵
def cm_plot(y,yp):
    from sklearn.metrics import confusion_matrix 
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',
                         verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    return plt

"""
数据预处理
"""
data = pd.read_excel(r"C:\Users\Administrator\Desktop\LR\贷款用户数据.xls")
data.head()

"""
数据标准化：
1. 0-1标准化： y = (xi-min{xj}) / max{xj} - min{xj}
2.Z标准化  yi = xi - x_average / s
"""
#对原始数据集变量与标签分离
X_whole = data.drop('还款拖欠情况', axis=1)
y_whole = data.还款拖欠情况

"""
切分数据集
"""
from sklearn.model_selection import train_test_split

x_train_w, x_test_w, y_train_w, y_test_w = \
    train_test_split(X_whole, y_whole, test_size = 0.2, random_state = 0)

"""
执行交叉验证操作
scoring:可选“accuracy”（精度）、recall（召回率）、roc_auc（roc值）
        neg_mean_squared_error（均方误差）、
  K折交叉验证
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#交叉验证选择较优惩罚因子
scores = []
c_param_range = [0.01,0.1,1,10,100]
z = 1
for i in c_param_range:
    start_time = time.time()
    lr = LogisticRegression(C = i, penalty = 'l2', solver='lbfgs')    
    score = cross_val_score(lr, x_train_w, y_train_w, cv=10, scoring='recall')
    score_mean = sum(score)/len(score)
    scores.append(score_mean)
    end_time = time.time()
    print("第{}次...".format(z))
    print("time spend:{:.2f}".format(end_time - start_time))
    print("recall值为:{}".format(score_mean))
    z +=1

best_c = c_param_range[np.argmax(scores)]
print()
print("最优惩罚因子为: {}".format(best_c))

"""
建立最优模型
"""
lr = LogisticRegression(C = best_c, penalty = 'l2', solver='lbfgs')
lr.fit(x_train_w, y_train_w)

"""
训练集预测
"""
from sklearn import metrics
#训练集预测概率
train_predicted_pr = lr.predict_proba(x_train_w)
train_predicted = lr.predict(x_train_w)
print(metrics.classification_report(y_train_w, train_predicted))
cm_plot(y_train_w, train_predicted).show() 


"""
测试集预测
"""
#预测结果
test_predicted = lr.predict(x_test_w)
#绘制混淆矩阵
print(metrics.classification_report(y_test_w, test_predicted))
cm_plot(y_test_w, test_predicted).show() 








"""
KNN 算法
 临近算法，（K-NearestNeighbor)
  K个最近的邻居
  擅长线性回归、分类
  最简单的算法
  优点： 无需训练；适合对稀有事件进行分类；
        对异常值不敏感
  缺点： 样本容量大的时候，计算时间很长
         不均衡样本效果较差
         

"""
import pandas as pd 

#读取数据
"""
train_data:训练集
test_data:测试集
"""
train_data = pd.read_excel("鸢尾花训练数据.xlsx",encoding='utf8')
test_data = pd.read_excel("鸢尾花测试数据.xlsx",encoding='utf8')

"""
处理训练集数据;
数据重排;变量与标签分离.
"""
#train_data = train_data.sample(frac=1) 
train_data.columns
train_X = train_data[['萼片长(cm)', '萼片宽(cm)', '花瓣长(cm)', '花瓣宽(cm)']]
train_y = train_data[['类型_num']]

"""
标准化语法
Z-Score标准化
"""
from sklearn.preprocessing import scale

data = pd.DataFrame()
data['萼片长标准化'] = scale(train_X['萼片长(cm)'])
data['萼片宽标准化'] = scale(train_X['萼片宽(cm)'])
data['花瓣长标准化'] = scale(train_X['花瓣长(cm)'])
data['花瓣宽标准化'] = scale(train_X['花瓣宽(cm)'])

"""
使用sklearn库中的KNN模块
"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)  # n_neighbors 默认是5
knn.fit(data, train_y)

knn.score(data, train_y)

train_predicted = knn.predict(data)
"""
绘制混淆矩阵;可视化混淆矩阵.
"""
from sklearn import metrics

#绘制混淆矩阵
print(metrics.classification_report(train_y, train_predicted))

#可视化混淆矩阵
def cm_plot(y,yp):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',
                         verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    return plt
        
cm_plot(train_y, train_predicted).show() 


"""
使用测试集数据进行测试
"""
test_X = test_data[['萼片长(cm)', '萼片宽(cm)', '花瓣长(cm)', '花瓣宽(cm)']]
test_y = test_data[['类型_num']]

"""
标准化语法
Z-Score标准化
"""
from sklearn.preprocessing import scale

data_test = pd.DataFrame()
data_test['萼片长标准化'] = scale(test_X['萼片长(cm)'])
data_test['萼片宽标准化'] = scale(test_X['萼片宽(cm)'])
data_test['花瓣长标准化'] = scale(test_X['花瓣长(cm)'])
data_test['花瓣宽标准化'] = scale(test_X['花瓣宽(cm)'])

#预测结果
test_predicted = knn.predict(data_test)
#预测概率
test_predicted_pr = knn.predict_proba(data_test)

"""
绘制混淆矩阵;可视化混淆矩阵.
"""
#可视化混淆矩阵
cm_plot(test_y, test_predicted).show() 






######   多分类问题  
"""
KNN算法对女性约会对象分类

"""
import pandas as pd

#可视化混淆矩阵
def cm_plot(y,yp):
    from sklearn.metrics import confusion_matrix 
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',
                         verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    return plt

"""
数据预处理
1:不喜欢
2：一般魅力
3：极具魅力
"""
data = pd.read_table("dating.txt")
data.head()

data.columns
"""
数据标准化：Z标准化
"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['每年获得的飞行常客里程数'] = scaler.fit_transform(data[['每年获得的飞行常客里程数',]])
data['玩视频游戏所消耗的时间百分比'] = scaler.fit_transform(data[['玩视频游戏所消耗的时间百分比',]])
data['每周消费的冰淇淋公升数'] = scaler.fit_transform(data[['每周消费的冰淇淋公升数',]])
data.head()


"""
切分数据集
"""
from sklearn.model_selection import train_test_split

X_whole = data.drop('类别', axis=1)
y_whole = data.类别
x_train_w, x_test_w, y_train_w, y_test_w = \
    train_test_split(X_whole, y_whole, test_size = 0.2, random_state = 0)

"""
使用sklearn库中的KNN模块
"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(x_train_w, y_train_w)

knn.score(x_train_w, y_train_w)

train_predicted = knn.predict(x_train_w)

"""
绘制混淆矩阵;可视化混淆矩阵.
"""
from sklearn import metrics

#绘制混淆矩阵
print(metrics.classification_report(y_train_w, train_predicted))

cm_plot(y_train_w, train_predicted).show() 

"""
使用测试集数据进行测试
"""
#预测结果
test_predicted = knn.predict(x_test_w)
#绘制混淆矩阵
print(metrics.classification_report(y_test_w, test_predicted))
cm_plot(y_test_w, test_predicted).show() 

















"""
决策树算法
    概念： 通过对训练样本的学习，建立分类规则，根据规则，
           对新样本数据进行分类预测，属于有监督学习。
    核心： 所有数据从根节点一步一步落到叶子节点
    决策书树分类标准：
          1.ID3算法--熵值：随机变量的不确定性的度量，
                          或者说是物体内部的混乱程度
                       信息增益   
             
          2.C4.5算法  信息增益比
          3.CART决策树   基尼系数（越小越好）
    决策树的剪枝： 防止过拟合--泛化差
                   预剪枝和后剪枝
                   一般用预剪枝：
                       1.限制树的深度
                       2.限制叶子节点的个数以及叶子节点的样本数
                       3.基尼系数
         

"""
import pandas as pd

#可视化混淆矩阵
def cm_plot(y,yp):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',
                         verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    return plt


#导入数据
datas = pd.read_excel("电信客户流失数据.xlsx",encoding='utf8')
#将变量与结果划分开
data = datas.ix[:,:-1]
target = datas.ix[:,-1]

#划分数据集
"""
导入模块对数据进行划分；
"""
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = \
    train_test_split(data, target, test_size = 0.2, 
                     random_state = 42)
#定义决策树  
from sklearn import tree 
   
dtr = tree.DecisionTreeClassifier(criterion='gini', max_depth = 6,
                                  random_state = 42)
dtr.fit(data_train, target_train)

"""
训练集混淆矩阵
"""
#训练集预测值
train_predicted = dtr.predict(data_train)

from sklearn import metrics
#绘制混淆矩阵
print(metrics.classification_report(target_train, train_predicted))
#可视化混淆矩阵
cm_plot(target_train, train_predicted).show() 


"""
测试集混淆矩阵
"""
#测试集预测值
test_predicted = dtr.predict(data_test)

from sklearn import metrics
#绘制混淆矩阵
print(metrics.classification_report(target_test, test_predicted))
#可视化混淆矩阵
cm_plot(target_test, test_predicted).show() 
#对决策树测试集进行评分
dtr.score(data_test, target_test)

#要可视化显示 
"""
修改dtr为自己的变量名；
修改feature_names为自己的数据
最终生成一个.dot文件
"""
dot_data = \
    tree.export_graphviz(
        dtr,
        out_file = None,
        feature_names = data.columns,
        filled = True,
        impurity = False,
        rounded = True
    )
#导入pydotplus库解读.dot文件
"""
只用修改颜色"#FFF2DD"
"""
import pydotplus

graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")
               
from IPython.display import Image
Image(graph.create_png())

#导出决策树的图
graph.write_png("dtr.png")




#####   解决遗留问题  
#  1.决策树显示异常问题
#  2.过拟合问题    

import pandas as pd

#可视化混淆矩阵
def cm_plot(y,yp):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',
                         verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    return plt


#导入数据
datas = pd.read_excel("电信客户流失数据2.xlsx",encoding='utf8')
#将变量与结果划分开
data = datas.ix[:,:-1]
target = datas.ix[:,-1]

#划分数据集
"""
导入模块对数据进行划分；
"""
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = \
    train_test_split(data, target, test_size = 0.2, 
                     random_state = 0)
#定义决策树  
from sklearn import tree 
   
dtr = tree.DecisionTreeClassifier(criterion='gini', max_depth = 10,
                                  min_samples_leaf=5,random_state = 0)
dtr.fit(data_train, target_train)

"""
训练集混淆矩阵
"""
#训练集预测值
train_predicted = dtr.predict(data_train)

from sklearn import metrics
#绘制混淆矩阵
print(metrics.classification_report(target_train, train_predicted))
#可视化混淆矩阵
cm_plot(target_train, train_predicted).show() 


"""
测试集混淆矩阵
"""
#测试集预测值
test_predicted = dtr.predict(data_test)

from sklearn import metrics
#绘制混淆矩阵
print(metrics.classification_report(target_test, test_predicted))
#可视化混淆矩阵
cm_plot(target_test, test_predicted).show() 
#对决策树测试集进行评分
dtr.score(data_test, target_test)

#要可视化显示 
"""
修改dtr为自己的变量名；
修改feature_names为自己的数据
最终生成一个.dot文件
"""
dot_data = \
    tree.export_graphviz(
        dtr,
        out_file = None,
        feature_names = data.columns,
        filled = True,
        impurity = False,
        rounded = True
    )
#导入pydotplus库解读.dot文件
"""
只用修改颜色"#FFF2DD"
"""
import pydotplus

graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")
               
from IPython.display import Image
Image(graph.create_png())

#导出决策树的图
graph.write_png("dtr.png")





"""
决策树 之  回归树模型
     解决回归问题的决策模型即为回归树
     特点：必须是二叉树
     
         

"""

import pandas as pd
from sklearn import tree

#读取数据
data = pd.read_csv(r"C:\Users\wwb\Desktop\data.csv")
#print(data)

#变量与标签的分离
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
#print(x)
#print(y)

#实例化一个回归树对象
reg = tree.DecisionTreeRegressor(max_depth=2)
reg = reg.fit(x,y)

#预测
y_pr = reg.predict(x)

#可视化展示
import matplotlib.pyplot as plt

#绘制子图
plt.subplot(2,2,1)
plt.scatter(x,y,c='black')
plt.subplot(2,2,2)
plt.scatter(x,y_pr,c='r')
plt.subplot(2,2,3)
plt.scatter(x,y,c='black',s=100)
plt.scatter(x,y_pr,c='r',s=50)
plt.show()










"""
集成学习：将多个基学习器进行组合，来实现比
          单一学习显著优越的学习性能  
          基学习器：其实就是一个决策树
  集成学习的代表：
          1.bagging方法：
            典型的是随机森林（RandomForestClassifier）
                 (1) 数据采样随机
                 (2) 特征选取随机
                （3） 森林
                （4） 基分类器为决策树--基尼系数
             随机森林生成步骤：
                 原始数据集--随机样本 1-n--训练样本 1-n
                 --分别进行分类训练--分类器c1-cn --
                 投票--强分类器
             随机森林优点：
                 准确率极高
                 抗噪能力强、不容易过拟合
                 很容易处理高维的数据，不用做特征选择
                 容易实现并行化计算
             随机森林缺点：
                  决策树很多的时候，训练需要的空间和时间会较大
                  有许多不好解释的地方
                  
            2.boosting方法：典型的是Xgboost
            3.stacking方法：堆叠模型
  集成学习的应用：
           分类问题
           回归问题
           特征选取问题
           ...

           
     
         

"""


######   随机森林算法案例  垃圾邮件特征选取以及分类
"""
此数据库包含有关4597条电子邮件的信息.任务是确定给定的电子邮件是否是垃圾
邮件(类别1)，取决于其内容。

大多数属性表明某个特定的单词或字符是否经常出现在电子邮件中。

以下是属性的定义：
-48个连续的实属性，类型为word_freq_“word”=与“word”匹配的电子邮件中
单词的百分比。在这种情况下，“Word”是由非字母数字字符或字符串结尾的任
何字母数字字符组成的字符串。
-6个连续的实属性char_freq_“char”=与“char”匹配的电子邮件中字符的百分比。
-1连续实属性类型：Capital_Run_Length_Average=不间断大写字母序列的平均长度。
-1连续整数属性，类型为Capital_Run_Length=最长不间断大写字母序列的长度。
-1连续整数属性，类型为Capital_Run_Length_Total=电子邮件中大写字母的总数。
"""
import pandas as pd

#可视化混淆矩阵
def cm_plot(y,yp):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',
                         verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    return plt

"""
数据读取与划分
"""
df = pd.read_csv('spambase.csv')

from sklearn.model_selection import train_test_split
#数据划分
y = df.ix[:,-1]   #将标签列置于此
X = df.ix[:,:-1]   #删除标签列即可

xtrain, xtest, ytrain, ytest =\
    train_test_split(X, y, test_size=0.2, random_state=100)

#进行预测分析
"""
n_estimators:决策树的个数
max_features:特征的个数
此处可以根据需要设置每棵决策树的深度以及最小叶子节点的样本数等
"""
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_features=0.8,
    random_state=0
)
rf.fit(xtrain, ytrain)

#预测训练集结果
train_predicted = rf.predict(xtrain)
"""
训练集结果
"""
from sklearn import metrics
#绘制混淆矩阵
print(metrics.classification_report(ytrain, train_predicted))

#可视化混淆矩阵
cm_plot(ytrain, train_predicted).show() 

"""
测试集结果
"""
#预测测试集结果
test_predicted = rf.predict(xtest)

from sklearn import metrics
#绘制混淆矩阵
print(metrics.classification_report(ytest, test_predicted))

#可视化混淆矩阵
cm_plot(ytest, test_predicted).show() 

"""
绘制特征重要程度排名
"""
import matplotlib.pyplot as plt
from pylab import mpl

importances = rf.feature_importances_
im = pd.DataFrame(importances)
clos = df.columns.values.tolist()
clos = clos[0:-1]
im['clos'] = clos
im = im.sort_values(by=[0], ascending=False)[:10]

#设置中文字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
#解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False 
index = range(len(im))
plt.yticks(index, im.clos)
plt.barh(index, im[0])



















"""
贝叶斯算法
    贝叶斯（1702-1761） 英国数学家
    
    P（A|B） = (P(B|A)/P(B)) *P(A)
    P（A|B） P(B|A) 条件概率
    P(A) 先验概率
    bayes.MultinominalNB 多项式分布的朴素贝叶斯
    
    
"""
import pandas as pd

#可视化混淆矩阵
def cm_plot(y,yp):
    from sklearn.metrics import confusion_matrix 
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',
                         verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    return plt

"""
数据预处理
"""
data = pd.read_csv("iris.csv",encoding='utf8', engine='python',header=None)
data = data.drop(0, axis=1)
data.head()

#对原始数据集进行切分
X_whole = data.drop(5, axis=1)
y_whole = data[5]

"""
切分数据集
"""
from sklearn.model_selection import train_test_split

x_train_w, x_test_w, y_train_w, y_test_w = \
    train_test_split(X_whole, y_whole, test_size = 0.2, random_state = 0)

#导入朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
#实例化贝叶斯分类器
classifier = MultinomialNB(alpha=1)

#传入训练集数据
classifier.fit(x_train_w, y_train_w)

"""
训练集预测
"""
#绘制训练集混淆矩阵
train_pred = classifier.predict(x_train_w)
cm_plot(y_train_w, train_pred)

"""
测试集预测
"""
test_pred = classifier.predict(x_test_w)
cm_plot(y_test_w, test_pred)










"""
SVM算法 （Support Vector Machines）
   SVM：选择一个最佳的一条线（超平面）
   data--classifier--optimization--kerneling--hyperplane
   球--棍子--最大间隙--拍桌子--那张纸
   +1  0  -1
 两个衡量指标  
   1.点到超平面的远近 --确信度
   2.正确性：分类正确  yi*y(x)>0
 如何找到最优的超平面？
   1.找到离超平面最近的点
   2.最大化这个距离--使的离超平面最近的点
                    离超平面的距离越远越好
                    --拉格朗日乘子法
                    min(f(x))
                    约束条件：s.t.g(x)<=0
    引入松弛因子  C 惩罚因子越大越严格
 核函数：二维空间无法用一条直线分离开，映射到三维或者更高维的空间即可解决
      1.线性核函数  只能一种
      2.多项式核函数  只能2-3
      3.高斯核函数      都可以  γ值（我的理解是对应于方差）越小越好  拟合越好
SVM 优点：   有严格数学理论基础，可解释性强
             能找出对任务有关键影响的样本，即支持向量
             软间隔可以有效松弛目标函数
             核函数可以有效解决非线性问题
             能够避免“维度灾难”
             能在小样本训练集上能够得到比其他算法好很多的结果
     缺点：   大数据训练集难以实施 --官方建议超过1万  不适用SVM
              对参数的选择敏感 -- C 和 gammma 建议使用交叉验证来选择一个比较好的值
              支持向量多的话，预测计算的复杂程度高
 
    
"""


####  线性函数
"""
可视化SVM算法
"""
import pandas as pd

data = pd.read_csv(r"C:\Users\wwb\Desktop\iris.csv",header=None)

"""
可视化原始数据
"""
import matplotlib.pyplot as plt

data1 = data.iloc[:50,:]        
data2 = data.iloc[50:,:] 
#原始数据是四维，无法展示，选择两个进行展示
plt.scatter(data1[1],data1[3],marker='+')
plt.scatter(data2[1],data2[3],marker='o')

"""
使用SVM进行训练
"""
from sklearn.svm import SVC

X = data.iloc[:,[1,3]]
y = data.iloc[:,-1]
svm = SVC(kernel='linear',C=float('inf'),random_state=0)
svm.fit(X,y)

"""
可视化SVM结果
"""
#参数w[原始数据为二维数组]
w = svm.coef_[0]
#偏置项b[原始数据为一维数组]
b = svm.intercept_[0]
#超平面方程：w1x1+w2x2+b=0
#->>x2 = -(w1x1+b)/w2
import numpy as np

x1 = np.linspace(0,7,300)
#超平面方程
x2 = -(w[0]*x1+b)/w[1]
#上超平面方程
x3 = (1-(w[0]*x1+b))/w[1]
#下超平面方程
x4 = (-1-(w[0]*x1+b))/w[1]
#找到支持向量[二维数组]
vets = svm.support_vectors_

#可视化原始数据
plt.scatter(data1[1],data1[3],marker='+',color='b')
plt.scatter(data2[1],data2[3],marker='o',color='b')
#可视化超平面
plt.plot(x1,x2,linewidth=2,color='r')
plt.plot(x1,x3,linewidth=1,color='r',linestyle='--')
plt.plot(x1,x4,linewidth=1,color='r',linestyle='--')
#进行坐标轴限制
plt.xlim(4,7)
plt.ylim(0,5)
#可视化支持向量
plt.scatter(vets[0][0],vets[0][1],color='red')
plt.scatter(vets[1][0],vets[1][1],color='red')
plt.scatter(vets[2][0],vets[2][1],color='red')
plt.show()









"""
不同维度数据的可视化结果
惩罚因子：软间隔
"""
import pandas as pd

data = pd.read_csv(r"C:\Users\wwb\Desktop\iris.csv",header=None)

"""
可视化原始数据
"""
import matplotlib.pyplot as plt

data1 = data.iloc[:50,:]        
data2 = data.iloc[50:,:] 
#原始数据是四维，无法展示，选择两个进行展示
plt.scatter(data1[1],data1[4],marker='+')
plt.scatter(data2[1],data2[4],marker='o')

"""
使用SVM进行训练
"""
from sklearn.svm import SVC

X = data.iloc[:,[1,4]]
y = data.iloc[:,-1]
svm = SVC(kernel='linear',C=float(0.1),random_state=0)
svm.fit(X,y)

"""
可视化SVM结果
"""
#参数w[原始数据为二维数组]
w = svm.coef_[0]
#偏置项b[原始数据为一维数组]
b = svm.intercept_[0]
#超平面方程：w1x1+w2x2+b=0
#->>x2 = -(w1x1+b)/w2
import numpy as np

x1 = np.linspace(0,7,300)
#超平面方程
x2 = -(w[0]*x1+b)/w[1]
#上超平面方程
x3 = (1-(w[0]*x1+b))/w[1]
#下超平面方程
x4 = (-1-(w[0]*x1+b))/w[1]
#找到支持向量[二维数组]
vets = svm.support_vectors_

#可视化原始数据
plt.scatter(data1[1],data1[4],marker='+',color='b')
plt.scatter(data2[1],data2[4],marker='o',color='b')
#可视化超平面
plt.plot(x1,x2,linewidth=2,color='r')
plt.plot(x1,x3,linewidth=1,color='r',linestyle='--')
plt.plot(x1,x4,linewidth=1,color='r',linestyle='--')
#进行坐标轴限制
plt.xlim(4,7)
plt.ylim(0,2)
#可视化支持向量
plt.scatter(vets[:, 0], vets[:, 1], c='r', marker='x')
plt.show()






"""
高斯核函数的参数对超平面的影响
"""
from sklearn import svm
#生成数的模块儿
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

"""
超平面绘制函数
clf:SVM分类器
X:训练数据特征
y:训练数据标签
h:有关决策边界的绘制【设置为0.02即可】
draw_sv:是否绘制支持向量
title:图像的标题
"""
def plot_hyperplane(clf, X, y, 
                    h=0.02, 
                    draw_sv=True, 
                    title='hyperplan'):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y==label][:, 0], 
                    X[y==label][:, 1], 
                    c=colors[label], 
                    marker=markers[label])
    # 画出支持向量
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')

#生成数据
X, y = make_moons(n_samples=150, noise=0.15, random_state=42)

#实例化不同的svm分类器
clf_rbf1 = svm.SVC(C=1, kernel='rbf', gamma=0.01)
clf_rbf2 = svm.SVC(C=1, kernel='rbf', gamma=5)
clf_rbf3 = svm.SVC(C=100, kernel='rbf', gamma=0.01)
clf_rbf4 = svm.SVC(C=100, kernel='rbf', gamma=5)
clf_rbf5 = svm.SVC(C=10000, kernel='rbf', gamma=0.01)
clf_rbf6 = svm.SVC(C=10000, kernel='rbf', gamma=5)
#设置图像大小
plt.figure(figsize=(10, 10), dpi=144)

#做一个分类器列表
clfs = [ clf_rbf1, clf_rbf2, clf_rbf3, clf_rbf4, clf_rbf5, clf_rbf6]
#每个图像的标题
titles = ['C=1,gamma=0.01', 
          'C=1,gamma=5', 
          'C=100,gamma=0.01', 
          'C=100,gamma=5',
          'C=10000,gamma=0.01', 
          'C=10000,gamma=5']
#遍历所有的分类器，然后进行绘图展示
for clf, i in zip(clfs, range(len(clfs))):
    clf.fit(X, y)
    plt.subplot(3, 2, i+1)
    plot_hyperplane(clf, X, y, title=titles[i])




























"""
聚类算法
K-means 
   聚类：1.无监督问题：数据没有标签
         2.概念：相似的东西分为一组，也就是物以类聚，人以群分
    基本概念：聚成多少簇：K 需要自己设定
             距离的度量：一般用欧式距离
             质心：各向量的均值
             优化：簇内距离最小化；簇间距离最大化
 轮廓系数： 聚类效果的评价方式  [-1，1]  
           si接近1，则说明样本i聚类合理
           si接近-1，则说明样本i更应该分类到另外的簇
           si接近0，则说明样本i在两个簇的边界上   
 K-mean优缺点：
          优点：简单、快速、适合常规数据集
          缺点：K值难以确定
               很难发现任意形状的簇
不知道K值的情况下，最好使用DBSCAN算法
    
"""
#####  啤酒分类

import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics 

"""
可视化网站
https://www.naftaliharris.com/blog/visualizing-k-means-clustering/
"""
#读取文件
beer = pd.read_table("C:/Users/Administrator/Desktop/数据/K-means/data.txt",
                   sep=' ', encoding='utf8', engine='python')
#传入变量（列名）
X = beer[["calories","sodium","alcohol","cost"]]

#寻找合适的K值
"""
根据分成不同的簇，自动计算轮廓系数得分

"""
scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score) 
print(scores)    
#绘制得分结果
import matplotlib.pyplot as plt

plt.plot(list(range(2,20)), scores)
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")

#聚类
km = KMeans(n_clusters=2).fit(X)  #K值为3【分为3类】
beer['cluster'] = km.labels_
beer.sort_values('cluster')     #对聚类结果排序【排序时不能修改beer数据框，否则会与X中的数据对不上】

#对聚类结果进行评分
"""
采用轮廓系数评分
X:数据集   scaled_cluster：聚类结果
score：非标准化聚类结果的轮廓系数
"""
score = metrics.silhouette_score(X,beer.cluster)
print(score)  























"""
聚类算法
DBSCAN算法
      概念：基于密度的带噪声的空间聚类算法-
            -把簇定义为密度相连的点的最大集合，能够
            把高密度区域划分为簇
      实现过程：1.输入数据集
                2.指定半径
                3.指定密度阈值
                
    
"""
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics 

#读取文件
beer = pd.read_table("C:/Users/Administrator/Desktop/数据/K-means/data.txt",
                   sep=' ', encoding='utf8', engine='python')
#传入变量（列名）
X = beer[["calories","sodium","alcohol","cost"]]
#DBSCAN聚类分析
"""
eps:半径
min_samples：最小密度   【就是圆内最少有几个样本点】
labels：分类结果  【自动分类，-1为异常点】
"""
db = DBSCAN(eps=20, min_samples=3).fit(X)
labels = db.labels_

#添加结果至原数据框
beer['cluster_db'] = labels
beer.sort_values('cluster_db')

#查看分类均值
beer.groupby('cluster_db').mean()

#对聚类结果进行评分
"""
采用轮廓系数评分
X:数据集   scaled_cluster：聚类结果
score：非标准化聚类结果的轮廓系数
"""
score = metrics.silhouette_score(X,beer.cluster_db)
print(score) 


















"""
PCA (sklearn.decomposition.PCA)
主成分分析：通用的降维工具   数据的特征又叫数据的维度-
           -减少数据的特征即降维
  向量的内积：A*B = |A|*|B|*cos(α)
  基：(1  0)
      (0  1)
  最优基：方差最大方向
  协方差：表征两个字段之间的相关性，为0时  完全独立
  优缺点：
     优点：计算方法简单； 
           可以减少指标筛选； 
           消除变量之间的多重共线性；
           在一定程度上能够减少噪声数据；
     缺点：特征必须是连续变量；
            无法解释降维之后的数据是什么；
            贡献率小的成分有可能更重要

"""

####  测试

import numpy as np
from sklearn.decomposition import PCA

#生成数据
X = np.array([[1,1],[1,3],[2,3],[4,4],[2,4]])
#实例化PCA对象
pca = PCA(n_components=0.9)
#训练模型
pca.fit(X)
#查看降维后的数据
new = pca.fit_transform(X)
print(new)

-3/np.sqrt(2)  -2.1213203435596424
-1/np.sqrt(2)  -0.70710678118654746
0
2.1213203435596424
0.70710678118654746

[[ 2.12132034]
 [ 0.70710678]
 [-0.        ]
 [-2.12132034]
 [-0.70710678]]


#打印特征所占的百分比
pca.explained_variance_ratio_






####  实例化   PCA

from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#数据读取
data = pd.read_excel(".\hua.xlsx")

#数据划分
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

data.columns

#实例化PCA对象
pca = PCA(n_components=0.90)
#进行训练
pca.fit(X) 

print('特征所占百分比:{}'.format(sum(pca.explained_variance_ratio_)))
print(pca.explained_variance_ratio_) 
#print(pca.singular_values_)
print('PCA降维后数据:')
new_x = pca.fit_transform(X)
print(new_x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(new_x, y, test_size = 0.2, random_state = 0)


#导入逻辑回归分类器
from sklearn.linear_model import LogisticRegression
#实例化逻辑回归分类器
classifier = LogisticRegression()

#传入训练集数据
classifier.fit(x_train, y_train)


#可视化降维后结果
red_x,red_y=[],[]
blue_x,blue_y=[],[]

for i in range(len(new_x)):
    if y[i] ==0:
        red_x.append(new_x[i][0])
        red_y.append(new_x[i][1])
    elif y[i]==1:
        blue_x.append(new_x[i][0])
        blue_y.append(new_x[i][1])

#可视化
def x2(x1):
    print('coef_:[w1]',classifier.coef_ ,classifier.coef_[0][0])
    print("intercept_[w0]:",classifier.intercept_)
    print("coef_ shape:",classifier.coef_.shape)
    print("intercept_shape:",classifier.intercept_.shape)
    print("classes_:",classifier.classes_)
    #w1x1 + w2x2 + w0 = 0【当这个值大于0时，属于1；当这个值小于0时，属于0】
    #(-w1*x1 - w0)/w2
    return (-classifier.coef_[0][0] * x1 - classifier.intercept_[0]) / classifier.coef_[0][1]

plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
x1_plot = np.linspace(-2, 2,100)
x2_plot = x2(x1_plot)
plt.plot(x1_plot, x2_plot)
plt.show()

#可视化混淆矩阵
def cm_plot(y,yp):
    from sklearn.metrics import confusion_matrix 
    
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',
                         verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    return plt

"""
训练集预测
"""
#绘制训练集混淆矩阵
train_pred = classifier.predict(x_train)
cm_plot(y_train, train_pred)

"""
测试集预测
"""
test_pred = classifier.predict(x_test)
cm_plot(y_test, test_pred)




















"""
SVD奇异值分解
    应用：1.数据降维
          2.推荐算法
          3.自然语言处理
          4.图像压缩
    A = U*Σ*（V转置）
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def pic_compress(k, pic_array):
    global u,sigma,vt,sig,new_pic
    
    u, sigma, vt = np.linalg.svd(pic_array)
    sig = np.eye(k) * sigma[: k]
    new_pic = np.dot(np.dot(u[:, :k], sig), vt[:k, :])  # 还原图像
    size = u.shape[0] * k + sig.shape[0] * sig.shape[1] + k * vt.shape[1]  # 压缩后大小
    return new_pic, size

filename = r"C:\Users\Administrator\Desktop\timg.jpg"
ori_img = np.array(Image.open(filename))
new_img, size = pic_compress(100, ori_img)
print("original size:" + str(ori_img.shape[0] * ori_img.shape[1]))
print("compress size:" + str(size))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(ori_img,cmap='gray')
ax[0].set_title("before compress")
ax[1].imshow(new_img,cmap='gray')
ax[1].set_title("after compress")
plt.show()















"""
python 文字处理


"""
### 去除空格
str1 = " 今天我要玩QQ飞车，明天要去吃火锅 "

print(str1)
# 去除前后空格
print(str1.strip())
#去除右空格
print(str1.rstrip())
#去除左空格
print(str1.lstrip())


### 去除字符
str1 = "Q今天我要玩QQ飞车，明天要去吃Q火锅 Q"

print(str1)
# 去除前后指定字符
print(str1.strip("Q"))
#去除右字符
print(str1.rstrip("Q"))
#去除左字符
print(str1.lstrip("Q"))




### 字符替换
str1 = "今天我要玩QQ飞车，今天要去吃火锅"
print(str1)

# 替换为其他字符
print(str1.replace("今天", "明天"))

# 去除某个字符
print(str1.replace("今天", ""))


### 分割于合并字符串
str1 = "今天 我 要 玩 QQ飞车，今天 要 去吃 火锅"

# 分割字符串--列表
str1_splited = str1.split(" ")
print(str1.split(" "))

# 合并字符串
str1_join = "".join(str1_splited)
print(str1_join)


"""
中文分词工具
jieba库
1.是什么？
 一种分词工具
原句：今天我要玩QQ飞车，今天要去吃火锅
分词：今天 我 要 玩 QQ飞车，今天 要 去吃 火锅
免费的  做的比较好的
2.为什么要用
计算机不认识文字  计算机只认识 0 1 
3.怎么用？
1.自己安装  pip install jieba 
2.官网自己安装  半自动
"""
import jieba
"""
分词
"""
# 全模式分词
str1 = "我叫赵凯,我爱心理学，希望可以有所成就"
result = jieba.cut(str1,cut_all = True)
print("全模式:"+"-".join(result))
# 精确模式分词
str1 = "我叫赵凯,我爱心理学，希望可以有所成就"
result = jieba.cut(str1,cut_all = False)
print("精确模式:"+"-".join(result))


# 默认的分词模式时精确模式
str2 = "九阳神功和黯然销魂掌"
for w in jieba.cut(str2):
    print(w)
# 添加自定义词组
jieba.add_word("九阳神功")
jieba.add_word("黯然销魂掌")
# 分词打印
for w in jieba.cut(str2):
    print(w)
# 添加自定义字典[txt文件,utf-8编码]
jieba.load_userdict("path")




"""
关键词提取
  小说、新闻、论文
  1.数据收集，建立相应的语料库
  2.数据准备：导入分词库和停用词库
  3.模型建立：使用jieba分词，对语料库进行分词处理
  4.模型结果统计：根据分词结果，进行词频统计，并绘制词云图
  5.TF-IDF分析：得到加权后的分词结果
              TF：词频
              IDF：逆向文档频率--越大，词条类别区分能力越强
              TF*IDF---就相当于加权
 余弦相似性：余弦值越接近1，就表明夹角越接近0--也就是两个向量越相似             
"""
import jieba.analyse
text = """对北京各区进行流行病学调查（以下简称流调）的人来说，过去的这个端午节是四个字：随时待命。在这次新发地批发市场聚集性疫情防控中，北京海淀区疾控中心传染病地方病控制科李洋所在的流调组由3个流调队轮流当班，“忙起来3天睡不到10个小时”。

　　相比于对病毒溯源，流调人员做的首先是对病例溯源。距新发地批发市场被锁定为传播途径已经两周，与新发地相关的“第一代病例”的密切接触者仍在隔离观察。在勾勒疫情传播链条上，流调人员小心翼翼地在病例活动轨迹中锁定疑点，核实每一种可能性。

　　李洋最近就将海淀区永定路70号院的一个公共厕所确认为6月21日一对确诊夫妇的感染途径。对于“破案”过程，他直言没那么曲折，“流调第1天就提出了假设，之后4天都在排查其他可能、最终核实假设”。

　　这对在海淀永定路天客隆超市经营烤肉拌饭档口夫妇的流调结果显示，这次永定路天客隆超市疫情，是从新发地商户关联到了与之密切接触的玉泉东市场商户，再关联到玉泉东市场商户居住的永定路70号院，最终关联到去这里上卫生间的天客隆超市烤肉拌饭店老板。

　　不放过任何一个可能性

　　对李洋和同事来说，从“战时”到“平时”的转化是突然发生的。6月6日零时，北京市突发公共卫生事件响应级别调整至三级。6月11日，北京首个新增本地新冠肺炎确诊病例“西城大爷”出现，北京海淀区疾控中心的流调人员当天接到通知，调查与“西城大爷”在海淀区的密切接触者。这之后，海淀区也相继出现确诊患者。

　　相比于未发病的密切接触者，流调人员对确诊患者的调查更为细致，他们需要让患者回忆出发病前14天内的活动轨迹。

　　“记忆出现模糊是常有的，尤其是年纪大的患者，要回忆14天很困难。”李洋记得，流调人员在核实“西城大爷”回忆在海淀区内的几个活动点时，就用了些辅助“证据”，比如，用支付记录来验证他的确到过某些地方。

　　对李洋和同事来说，流调工作的一个重点在于排查所有可能性，“和春节期间关注输入病例不同，这次北京疫情要关注批发市场和可能被污染的物品，重视环境样本”。

　　在调查6月21日确诊的天客隆超市二楼美食城女摊主时，李洋注意到了该患者自己提出的疑点——“永定路70号院”，该地点此前曾4次出现在通报确诊病例活动场所中，本就是李洋和同事关注的“可疑场所”。

　　“她回忆6月12号去过永定路70号院520号楼的公共厕所，因为当天超市停电停水，地下一层的厕所不方便使用，当时永定路70号院也没有出现病例，后来她从新闻中看到那里有确诊病例，在接受流调时就说出了这个怀疑。”李洋说，流调的第一天，他和同事就从这个疑点出发，作出了有关感染途径的初步假设，对于该患者而言，既没有新发地市场接触史，又没有接触过已知病人，“只能从其他感染途径来确定感染源”。

　　虽然有了假设，但为了排查其他可能性，李洋和同事仍旧将“能想到的点都查了一遍”。他们从该患者的摊位查起，对操作间冰箱里的剩余食材进行了采样，对工作环境也进行了物表采样，还了解了店铺的货源源头，“电话联系了供货商，由北京房山和山东的厂商直接供货，不存在传染可能性”。

　　到永定路70号院520号楼公共厕所的环境样本中发现核酸检测为阳性后，李洋与同事再次向该患者核实活动轨迹，最终确定了最初假设，锁定该公共厕所为患者感染途径。

　　更细化的流调模板

　　“流调的工作很‘烧脑’，考虑要全面，有时候初步流调会遗漏一些细小的点，需要再补充两到三次流调。”从事流调工作近20年，李洋笑称这份工作既需要逻辑推理能力，又需要在平时就积累相应的专业知识。在2003年“非典”期间，他就在疫情早期参与过现场流调工作。

　　十多年过去，针对不同的传染病，李洋与同事使用过不同的流调模板，针对同一种传染病的流调模板也会按情况更新。“我们现在使用的对新冠肺炎患者的流调模板，就和年初时不同，有十页左右，很多部分比之前的版本更细化，比如在临床指标里增加了正常值”。

　　更细化的问题意味着流调人员要更耐心与细心，“在提问上没有什么更好的办法，只能慢慢启发，让患者回忆”。在记忆模糊之外，患者的情绪有时也会影响流调进度，李洋观察到，在这次流调中，有的患者在得知核酸检测结果为阳性后反而情绪稳定下来，更加配合，“在之前等结果时会出现情绪波动，觉得不会是自己（确诊），不太配合”。

　　相比平日对普通传染病的流调工作，这次北京疫情的流调工作周期更长，强度更大，被称为“最严格的流调”。

　　“这次每个流调报告从接手到初步完成，基本上要经历两到三天，而且是连续工作，但平时对普通传染病的流调，8小时基本上就能搭好报告框架、丰富具体信息。”李洋说，这次需要调查的感染源很多，有的病例流调还要跨区协助调查。

　　在6月20日确诊的一位来自海淀区八里庄街道的男性病例，就通过协助调查，最终确定了感染途径。该病例没有新发地接触史，也不是确诊患者的密切接触者，但发病前曾在北京郊区的一家餐馆和另外一名确诊病例共同待过十几分钟，二者之间并不认识。

　　李洋与同事们仍在随时待命，“医院一给我们报告病例，我们就去流调”。

　　中国青年报客户端北京6月28日电"""
result = jieba.analyse.extract_tags(text,topK=6)
print("关键词："+" ".join(result))




#####   红楼梦分析
"""
分词：
1.读取文件 + 分词
"""
import pandas as pd
import numpy as np
import time
import os
import os.path
import codecs
import jieba

"""
1.读取文件，并将目录下的文件以数据框的形式存储

os.walk('path')函数:
每次遍历的对象都是返回一个三元组（root, dirs, files）
1.root:所指的是当前正在遍历的这个文件夹本身的地址
2.dirs:是一个list,内容是该文件夹中所有目录的名字(不包括子目录)
3.files:同样是list,内容是该文件夹中所有的文件(不包括子目录)
"""        
filePaths = []
fileContents = []
for root, dirs, files in os.walk(
    r"C:\Users\Administrator\Desktop\红楼梦"
):
    for name in files:
        filePath = os.path.join(root, name);
        filePaths.append(filePath);
        f = codecs.open(filePath, 'r', 'utf8')
        fileContent = f.read()
        f.close()
        fileContents.append(fileContent)
corpos = pd.DataFrame({
    'filePath': filePaths, 
    'fileContent': fileContents
})

"""
导入分词库,写入文件路径即可
文件格式为txt
"""
jieba.load_userdict(
r"C:\Users\Administrator\Desktop\new\红楼梦分析\红楼梦词库.txt")

"""
导入停用词库
"""
stopwords = pd.read_csv(
r"C:\Users\Administrator\Desktop\new\红楼梦分析\StopwordsCN.txt", 
encoding='utf8', engine='python',index_col=False)


"""
2.进行分词,并与停用词表进行对比删除,
然后进行词频统计。
segments：分词后的结果
"""
start_time = time.time()
segments = []
filePaths = []
for index, row in corpos.iterrows():
    filePath = row['filePath']
    fileContent = row['fileContent']
    segs = jieba.cut(fileContent)
    for seg in segs:
        if seg not in stopwords.stopword.values and len(seg.strip())>0:
            segments.append(seg)
            filePaths.append(filePath)

segmentDataFrame = pd.DataFrame({
    'segment': segments, 
    'filePath': filePaths
})
end_time = time.time()
print("分词耗时:{}.".format(end_time - start_time))


"""
进行词频统计
"""       
segStat = segmentDataFrame.groupby(
            by="segment"
        )["segment"].agg({
            "计数":np.size
        }).reset_index().sort(
            columns=["计数"],
            ascending=False
        )
segStat.index = range(0, len(segStat))

"""
绘制词云图
"""
from scipy.misc import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator

wordcloud=WordCloud(font_path="simhei.ttf",
                    background_color="white",
                    max_font_size=80)
word_frequence = {x[0]:x[1] for x in segStat.head(200).values}
wordcloud=wordcloud.fit_words(word_frequence)
plt.axis("off")
plt.imshow(wordcloud)

"""
3.美化词云图
"""
#添加图片
bimg = imread(
r"C:\Users\Administrator\Desktop\new\timg.png")
#设置对象参数
wordcloud = WordCloud(
background_color="white", mask=bimg, 
font_path="simhei.ttf",max_font_size=80
)
word_frequence = {x[0]:x[1] for x in segStat.head(200).values}
wordcloud=wordcloud.fit_words(word_frequence)

plt.figure(
    num=None, 
    figsize=(8, 6), dpi=80, 
    facecolor='w', edgecolor='k')

#设置文字颜色【与选中的图片颜色一致】
bimgColors = ImageColorGenerator(bimg)

#绘图展示
plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))
plt.show()

"""
4.关键词提取：
TF-IDF算法
导入jieba的分析功能,将文本的关键信息找出
topK:指提取出关键词的个数 
"""
import jieba.analyse

key_words= ",".join(segmentDataFrame['segment'])  
print ("  ".join(jieba.analyse.extract_tags
     (key_words,topK=11, withWeight=False)))

"""
TF-IDF算法的优点是简单快速，结果比较符合实际情况；
缺点是衡量标准不够全面，有时重要的词可能出现次数并不多。
而且，这种算法无法体现词的位置信息，出现位置靠前的词
与出现位置靠后的词，都被视为重要性相同，这是不正确的。
"""    














#定义输入文本函数
def input_path(path):
    import os
    import os.path
    import codecs
    
    file_Paths = []
    global file_Contents
    file_Contents = []
    for root, dirs, files in os.walk(path):
        for name in files:
            file_Path = os.path.join(root, name);
            file_Paths.append(file_Path);
            f = codecs.open(file_Path, 'r', 'utf8')
            file_Content = f.read()
            f.close()
            file_Contents.append(file_Content)
#定义分词函数
def segment_func():
    import jieba
    import numpy as np
    
    global segmentation
    global seg_count
    result_1 = []
    for index, row in corpus.iterrows():
        fileContent = row['file_Content']
        results = jieba.cut(fileContent)
        for result in results:
            if result not in stopwords.stopword.values and len(result.strip())>0:
                result_1.append(result)
    
    segmentation = pd.DataFrame({
        '分词结果': result_1, 
    })    
    seg_count = segmentation.groupby(by="分词结果")["分词结果"].agg({
                                                "计数":np.size})
    seg_count = seg_count.reset_index().sort(columns=["计数"],ascending=False)
#定义词云图绘制函数
def word_pic():
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    wordcloud=WordCloud(font_path="C:/Windows/Fonts/FZYTK.TTF",
                        background_color="white",max_font_size=50)
    word_frequence = {x[0]:x[1] for x in seg_count.head(200).values}
    wordcloud=wordcloud.fit_words(word_frequence)
    plt.axis("off")
    plt.imshow(wordcloud)

#定义美化词云图函数
def word_pic_b(path_b):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, ImageColorGenerator
    from scipy.misc import imread
    
    bimg = imread(path_b) 
    wordcloud=WordCloud(font_path="C:/Windows/Fonts/FZYTK.TTF",
                        background_color="white",max_font_size=150,mask=bimg)
    word_frequence = {x[0]:x[1] for x in seg_count.head(200).values}
    wordcloud=wordcloud.fit_words(word_frequence)
    plt.figure(
    figsize=(8, 6), dpi=100, 
    facecolor='w', edgecolor='b')
    bimgColors = ImageColorGenerator(bimg)
    plt.axis("off")
    plt.imshow(wordcloud.recolor(color_func=bimgColors))
    plt.show() 
#定义TF-IDF分析函数
def tfidf(nums):
    import jieba.analyse
    key_words = "".join(segmentation['分词结果'])  
    print ("  ".join(jieba.analyse.extract_tags(key_words,
                                                topK=nums, withWeight=False)))     

"""
1.输入数据路径
"""
import pandas as pd
path = "C:/Users/Administrator/Desktop/红楼梦"
input_path(path)
corpus = pd.DataFrame({'file_Content': file_Contents })  

"""
2.进行中文分词
"""
import jieba

#导入分词库
jieba.load_userdict(
r"C:\Users\Administrator\Desktop\new\红楼梦分析\红楼梦词库.txt")     
    
#导入停用词库    
stopwords = pd.read_csv(
r"C:\Users\Administrator\Desktop\new\红楼梦分析\StopwordsCN.txt", 
encoding='utf8', engine='python',index_col=False)  

#分词      
segment_func()   
    
"""
3.绘制词云图
"""    
word_pic()   

"""
4.美化词云图   
"""    
path_b = r"C:\Users\Administrator\Desktop\new\timg.png" 
word_pic_b(path_b)    
    
"""
5.TF-IDF分析
num: 输出关键词个数
"""   
nums = 10 
tfidf(nums)    




















"""
词向量转化小例子
"""
#从特征提取库中导入向量转化模块儿
from sklearn.feature_extraction.text import CountVectorizer
#需要转化的语句
"""
ngram_range(1, 2):对词进行组合
(1)本例组合方式：两两组合
['bird', 'cat', 'cat cat', 'cat fish', 'dog', 'dog cat', 'fish', 'fish bird']
(2)如果ngram_range(1, 3),则会出现3个词进行组合
['bird', 'cat', 'cat cat', 'cat fish', 'dog', 'dog cat', 'dog cat cat', 
'dog cat fish', 'fish', 'fish bird']
"""
texts=["dog cat fish","dog cat cat","fish bird", 'bird']

#实例化一个模型 
cv = CountVectorizer(ngram_range=(1,1))

#训练此模型
cv_fit=cv.fit_transform(texts)

#打印出模型的全部词库
print(cv.get_feature_names())

#打印出每个语句的词向量
print(cv_fit.toarray())

#打印出所有数据求和结果
print(cv_fit.toarray().sum(axis=0))





"""
新闻分类
"""
"""
1.数据读取与处理
"""
import pandas as pd

df_news = pd.read_table(
r"C:\Users\Administrator\Desktop\new\new.txt",
names=['category','theme','URL','content'],
encoding='utf-8')
#去除空值
df_news = df_news.dropna()
df_news.head()

#将content列数据取出并转化为list格式
contents = df_news.content.values.tolist()
print (contents[100])

"""
2.使用jieba分词
"""
import jieba
import time

start_time = time.time()
segments = []
for content in contents:
    results = jieba.lcut(content)
    if len(results) > 1 and results != '\r\n': #换行符
        segments.append(results)             
end_time = time.time()
print("分词耗时:{}.".format(end_time - start_time))

#分词结果储存在新的数据框中
df_results=pd.DataFrame({'segment':segments})
df_results.head()

"""
3.移除停用词
segments_clean:每一篇文章的分词结果
all_words：所有文章总的分词结果
"""
#导入停用词库
stopwords = pd.read_csv(
r"C:\Users\Administrator\Desktop\new\红楼梦分析\StopwordsCN.txt", 
encoding='utf8', engine='python',index_col=False)  

#定义去除停用词函数
def drop_stopwords(contents,stopwords):
    global segments_clean
    global all_words
    
    segments_clean = []
    all_words = []
    for content in contents:
        line_clean = []
        for word in content:
            if word in stopwords and len(word.strip())>0:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        segments_clean.append(line_clean)
    return segments_clean,all_words

#调用去除停用词函数
contents = df_results.segment.values.tolist()  #DataFrame格式转为list格式
stopwords = stopwords.stopword.values.tolist()  #停用词转为list格式
start_time = time.time()
contents_clean,all_words = drop_stopwords(contents,stopwords)
end_time = time.time()
print("分词耗时:{}.".format(end_time - start_time))

#分词结果储存在新的数据框中
df_segments_clean=pd.DataFrame({'segments_clean':segments_clean})
df_segments_clean.head()

"""
4.TF-IDF关键词提取
"""
import jieba.analyse

index = 2400
print (df_news['content'][index])
content_str = \
 "".join(df_segments_clean['segments_clean'][index])  
print ("  ".join(jieba.analyse.extract_tags(
        content_str, topK=5, withWeight=False)))

"""
5.朴素贝叶斯分类
"""
#分词后数据添加标签
df_train=pd.DataFrame({'segments_clean':segments_clean,
      'label':df_news['category']})
df_train.head()

df_train.label.unique()

#将标签数据数值化
label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, 
"体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping)
df_train.head()

#数据切分
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
train_test_split(df_train['segments_clean'].values, 
                 df_train['label'].values, random_state=1)

x_train[0][1]

#更改输入格式
"""
由于输入格式要求为List,而不是list of list 的格式,所以更改输入数据格式.
"""
words = []
for line_index in range(len(x_train)):
    words.append(' '.join(x_train[line_index]))
words[0]   

#导入将词汇转化为向量的库
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(analyzer='word', max_features=4000, 
                      lowercase = False, ngram_range=(1,4))
vec.fit(words)

#导入朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(words), y_train)

#测试集数据进行分析
test_words = []
for line_index in range(len(x_test)):
    test_words.append(' '.join(x_test[line_index]))
test_words[0]

#测试集数据预测得分
classifier.score(vec.transform(test_words), y_test)





















































