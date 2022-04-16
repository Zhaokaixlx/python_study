# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:55:04 2022

@author: Administrator
"""
# 导入库
import numpy as np
import pandas as pd
# 导入数据集
dataset = pd.read_csv(r"D:\pycharmcode\12\Data1.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
print("x")
print("y")
# 处理缺失的数据
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = "mean")
impter = SimpleImputer.fit(x[:,1:3])

