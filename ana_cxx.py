# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:54:10 2022

@author: zhaokai
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.stats import fdr_correction,f_mway_rm


"""
先搞ec
3*4
A*B
ec(1\2\3) * 节律波(4)
"""

# 读取不同条件下的数据--
# 可以用for循环  但是没有这个思路清晰

data_A1B1 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco1\ec1.α波.xlsx").set_index("File",inplace=True)
data_A1B2 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco1\ec1.β波.xlsx").set_index("File",inplace=True)
data_A1B3 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco1\ec1.δ波.xlsx").set_index("File",inplace=True)
data_A1B4 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco1\ec1.θ波.xlsx").set_index("File",inplace=True)
data_A2B1 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco2\ec2.α波.xlsx").set_index("File",inplace=True)
data_A2B2 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco2\ec2.β波.xlsx").set_index("File",inplace=True)
data_A2B3 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco2\ec2.δ波.xlsx").set_index("File",inplace=True)
data_A2B4 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco2\ec2.θ波.xlsx").set_index("File",inplace=True)
data_A3B1 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco3\ec3.α波.xlsx").set_index("File",inplace=True)
data_A3B2 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco3\ec3.β波.xlsx").set_index("File",inplace=True)
data_A3B3 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco3\ec3.δ波.xlsx").set_index("File",inplace=True)
data_A3B4 = pd.read_excel(r"C:\Users\Administrator\Desktop\cxx\eco3\ec3.θ波.xlsx").set_index("File",inplace=True)

# 首先对数据进行reshape方便后续合并
reshaped_A1B1 = data_A1B1.reshape(9, 1, 19)
reshaped_A1B2 = data_A1B2.reshape(9, 1, 19)
reshaped_A2B1 = data_A2B1.reshape(9, 1, 19)
reshaped_A2B2 = data_A2B2.reshape(9, 1, 19)
# 把数据按照两个因素的顺序（A1B1、A1B2、A2B1、A2B2）合并
data_combine = np.concatenate((reshaped_A1B1, reshaped_A1B2,
                               reshaped_A2B1, reshaped_A2B2), axis=1)
# 设置变量水平
factor_levels = [3, 4]
# 使用MNE的f_mway_rm函数进行2×2方差分析
# 变量A的主效应
f_main_A, p_main_A = f_mway_rm(data_combine, factor_levels, effects='A')
# 变量B的主效应
f_main_B, p_main_B = f_mway_rm(data_combine, factor_levels, effects='B')
# 交互效应
f_inter, p_interaction = f_mway_rm(data_combine, factor_levels, effects='A:B')
# FDR矫正
#rejects_A, p_main_A = fdr_correction(p_main_A, alpha=0.05)
#rejects_B, p_main_B = fdr_correction(p_main_B, alpha=0.05)
#rejects_inter, p_interaction = mne.stats.fdr_correction(p_interaction, alpha=0.05)
# 可视化经过矫正的统计检验结果
# 图片下方三行灰色竖线，有下至上分别代表A主效应、B主效应和交互效应显著的时间点
plt.plot(times, np.average(data_A1B1, axis=0), label='A1B1')
plt.plot(times, np.average(data_A1B2, axis=0), label='A1B2')
plt.plot(times, np.average(data_A2B1, axis=0), label='A2B1')
plt.plot(times, np.average(data_A2B2, axis=0), label='A2B2')
for i in range(1200):
    if p_main_A[i] < 0.05:
        plt.axvline(x=times[i], ymin=0.01, ymax=0.06, color='grey', alpha=0.2)
    if p_main_B[i] < 0.05:
        plt.axvline(x=times[i], ymin=0.07, ymax=0.12, color='grey', alpha=0.2)
    if p_interaction[i] < 0.05:
        plt.axvline(x=times[i], ymin=0.13, ymax=0.18, color='grey', alpha=0.2)
plt.legend()
plt.show()
