# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:23:33 2022

@author: zhaokai

"""
"""
20行的py代码进行原始数据的读取和源估计
"""
import mne
fname = r"D:\py_mne\MNE-sample-data\MEG\sample\sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(fname,preload = True) # load data
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels
raw.filter(l_freq=0.1, h_freq=30)  # low-pass filter
events = mne.find_events(raw, 'STI014')  # extract events and epoch data
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,
                   reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()  # compute evoked
evoked.plot()  # butterfly plot the evoked data
cov = mne.compute_covariance(epochs, tmax=0, method='shrunk')
fwd = mne.read_forward_solution(fname, surf_ori=True)
inv = mne.minimum_norm.make_inverse_operator(
     raw.info, fwd, cov, loose=0.2)  # compute inverse operator
stc = mne.minimum_norm.apply_inverse(
    evoked, inv, lambda2=1. / 9., method='dSPM')  # apply it
stc_fs = stc.morph('fsaverage')  # morph to fsaverage
stc_fs.plot()  # plot source data on fsaverage's brain







# 引入python库
import mne
# from mne.datasets import sample
import matplotlib.pyplot as plt

# 该fif文件存放地址
fname = r"D:\py_mne\MNE-sample-data\MEG\sample\sample_audvis_raw.fif"

raw = mne.io.read_raw_fif(fname)

# 根据type来选择 那些良好的MEG信号(良好的MEG信号，通过设置exclude="bads") channel,
# 结果为 channels所对应的的索引
picks = mne.pick_types(raw.info, meg=True, exclude='bads')
t_idx = raw.time_as_index([10., 20.])
data, times = raw[picks, t_idx[0]:t_idx[1]]
plt.plot(times,data.T)
plt.title("Sample channels")
plt.show()

"""
绘制SSP矢量图
"""
raw.plot_projs_topomap()
plt.show()


"""
绘制电极位置
"""
raw.plot_sensors()
plt.show()








"""
在此单被试分析篇中，按照脑电预处理流程，分为以下8个步骤：

Step 1: 读取数据
Step 2: 滤波
Step 3: 去伪迹
Step 4: 重参考
Step 5: 分段
Step 6: 叠加平均
Step 7: 时频分析
Step 8: 提取数据
  Python工具包包括： NumPy及MNE-Python
"""
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
# 数据地址（改成自己的数据地址，在EEGLAB文件夹的sample_data文件夹下）
data_path = "/Users/zitonglu/Desktop/EEG/eeglab14_1_2b/sample_data/eeglab_data.set"

# MNE-Python中对多种格式的脑电数据都进行了支持：
# *** 如数据后缀为.set (来自EEGLAB的数据)
#     使用mne.io.read_raw_eeglab()
# *** 如数据后缀为.vhdr (BrainVision系统)
#     使用mne.io.read_raw_brainvision()
# *** 如数据后缀为.edf
#     使用mne.io.read_raw_edf()
# *** 如数据后缀为.bdf (BioSemi放大器)
#     使用mne.io.read_raw_bdf()
# *** 如数据后缀为.gdf
#     使用mne.io.read_raw_gdf()
# *** 如数据后缀为.cnt (Neuroscan系统)
#     使用mne.io.read_raw_cnt()
# *** 如数据后缀为.egi或.mff
#     使用mne.io.read_raw_egi()
# *** 如数据后缀为.data
#     使用mne.io.read_raw_nicolet()
# *** 如数据后缀为.nxe (Nexstim eXimia系统)
#     使用mne.io.read_raw_eximia()
# *** 如数据后缀为.lay或.dat (Persyst系统)
#     使用mne.io.read_raw_persyst()
# *** 如数据后缀为.eeg (Nihon Kohden系统)
#     使用mne.io.read_raw_nihon()

# 读取数据
raw = mne.io.read_raw_eeglab(data_path, preload=True)


# 查看原始数据信息
print(raw)
print(raw.info)

#电极定位--可能需要手动导入脑电数据的电极位置信息
# locs文件地址
locs_info_path = "/Users/zitonglu/Desktop/EEG/eeglab14_1_2b/sample_data/eeglab_chan32.locs"
# 读取电极位置信息
montage = mne.channels.read_custom_montage(locs_info_path)
# 读取正确的导联名称
new_chan_names = np.loadtxt(locs_info_path, dtype=str, usecols=3)
# 读取旧的导联名称
old_chan_names = raw.info["ch_names"]
# 创建字典，匹配新旧导联名称
chan_names_dict = {old_chan_names[i]:new_chan_names[i] for i in range(32)}
# 更新数据中的导联名称
raw.rename_channels(chan_names_dict)
# 传入数据的电极位置信息
raw.set_montage(montage)

"""
当你的脑电电极位点为一些特定系统时，可以直接用mne.channels.make_standard_montage函数生成 以标准的国际10-20系统为例，对应代码即可改为：

montage = mne.channels.make_standard_montage("standard_1020")

MNE中现成的其他定位系统的montage可以通过以下网址查询：
https://mne.tools/stable/generated/mne.channels.make_standard_montage.html#mne.channels.make_standard_montage

"""

# 设定导联类型
# MNE中一般默认将所有导联类型设成eeg
# 将两个EOG导联的类型设定为eog
chan_types_dict = {"EOG1":"eog", "EOG2":"eog"}
raw.set_channel_types(chan_types_dict)

# 打印修改后的数据相关信息
print(raw.info)

# 可视化原始数据
# 绘制原始数据波形图
raw.plot(duration=5, n_channels=32, clipping=None)
# 绘制原始数据功率谱图
raw.plot_psd(average=True)
# 绘制电极拓扑图
raw.plot_sensors(ch_type='eeg', show_names=True)
# 绘制原始数据拓扑图
raw.plot_psd_topo()



"""
Step 2 滤波
     1.陷波滤波
        通过Step1中的功率谱图可以看到60Hz处可能存在环境噪音
        这里首先使用陷波滤波器去掉工频
        注意：在中国大陆及香港澳门地区（除台湾省以外）采集的数据一般工频会出现在50Hz处
        此例比较例外，切记通过功率谱图判断
     2.高低通滤波
          预处理步骤中，通常需要对数据进行高通滤波操作
        此处采用最常规的滤波操作，进行30Hz的低通滤波及0.1Hz的高通滤波
        高通滤波为了消除电压漂移，低通滤波为了消除高频噪音
"""
## 陷波滤波
raw = raw.notch_filter(freqs=(60))
# 绘制功率谱图
raw.plot_psd(average=True)


## 高低通滤波
raw = raw.filter(l_freq=0.1, h_freq=30)
"""
MNE中默认使用FIR滤波方法，若想使用IIR滤波方法，可通过修改参数method参数实现
默认method='fir'，使用IIR则修改为'iir'
对应代码即为：

raw = raw.filter(l_freq=0.1, h_freq=30, method='iir')
"""
# 绘制功率谱图
raw.plot_psd(average=True)



"""
Step 3 去伪迹
   1.去坏段:
          MNE中可以通过打开交互式数据地形图界面，手动进行坏段标记

            fig = raw.plot(duration=5, n_channels=32, clipping=None)
            fig.canvas.key_press_event('a')

            按a就可以打开这个GUI小窗口，add new label可以添加一个用于标记坏段的marker
            在MNE中，并不会将坏段直接删掉，而是进行了数据标记
            在之后的数据处理中，
            将进行数据处理的函数中的参数reject_by_annotation设为True即可在处理过程中自动排除标记的片段
            如果遇到GUI窗口无法弹出，需在脚本开头添加如下代码：

            import matplotlib
            matplotlib.use('TkAgg')

            注意：不推荐在Jupyter notebook中打开，容易卡死
     2.去坏道
            MNE中坏的导联也不是直接删掉，也是通过对坏道进行'bads'标记
            在这个例子中，假定导联'FC5'为坏道，则把'FC5'进行坏道标记
     3.独立成分分析（ICA）
            运行ICA
            MNE中进行ICA的编程思路是首先构建一个ICA对象（可以理解成造一个ICA分析器）
            然后用这个ICA分析器对脑电数据进行分析（通过ICA对象的一系列方法）
            由于ICA对低频分离效果不好
            这里对高通1Hz的数据进行ICA及相关成分剔除，再应用到高通0.1Hz的数据上



"""
## 去坏段
fig = raw.plot(duration=5, n_channels=32, clipping=None)
fig.canvas.key_press_event('a')

## 去坏道  假定导联'FC5'为坏道
# 坏道标记
raw.info['bads'].append('FC5')
# 打印出当前的坏道
print(raw.info['bads'])
# 如若'FC5'和'C3'都为坏道，则通过下述代码标记：
# raw.info['bads'].extend(['FC5', 'C3'])

## 运行ICA
ica = ICA(max_iter='auto')
raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)
ica.fit(raw_for_ica)
"""
这里没有设定n_components，即ICA的成分数让MNE的ICA分析器自动去选择
类似EEGLAB，如果希望ICA的成分数为固定个数，可以自定义设置（n_components<=n_channels）
以30个独立成分为例，对应代码改为如下即可：

ica = ICA(n_components=30, max_iter='auto')
"""
# 绘制各成分的时序信号图
ica.plot_sources(raw_for_ica)
# 绘制各成分地形图
ica.plot_components()
# 查看去掉某一成分前后信号差异
# 这里以去掉第2个成分（即ICA001）为例
ica.plot_overlay(raw_for_ica, exclude=[1])

# 单独可视化每个成分
# 这里可视化第2个成分（ICA001）和第17个成分（ICA016）
ica.plot_properties(raw, picks=[1, 16])

# 剔除成分

# 设定要剔除的成分序号
ica.exclude = [1]
# 应用到脑电数据上
ica.apply(raw)

# 绘制ICA后的数据波形图
raw.plot(duration=5, n_channels=32, clipping=None)





"""
Step 4 重参考
     由于此数据作者使用了乳突参考
     --若数据需要进行参考，以'TP9'和'TP10'为参考电极为例，可以使用以下代码：
      raw.set_eeg_reference(ref_channels=['TP9','TP10'])
     --若使用平均参考，则使用以下代码：
      raw.set_eeg_reference(ref_channels='average')
     --若使用REST参考，则使用以下代码：
     这里需要传入一个forward参数，详情可参考MNE对应介绍：https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html
      raw.set_eeg_reference(ref_channels='REST', forward=forward)
     --若使用双极参考，则使用以下代码： (这里'EEG X'和'EEG Y'分别对应用于参考的阳极和阴极导联)
      raw_bip_ref = mne.set_bipolar_reference(raw, anode=['EEG X'], cathode=['EEG Y'])

"""



"""
Step 5 分段
    1.提取事件信息
        首先，需要确定分段需要用到的markers
        查看数据中的markers
        而Events对象，则是数据分段需要用到的一种事件记录数据类型
        其用一个整型'Event ID'编码事件类型，以样本的形式来表示时间
        且不含有marker的持续时长，其内部数据类型为NumPy Array
    2.事件信息数据类型转换
        将Annotations类型的事件信息转为Events类型
    3.数据分段
        基于Events对数据进行分段
        这里提取刺激前1秒到刺激后2秒的数据，即'square' marker对应-1s到2s的数据
        取baseline时间区间为刺激前0.5s到刺激出现
        并进行卡阈值，即在epoch中出现最大幅值与最小幅值的差大于2×10^-4则该epoch被剔除
        注意：这里的阈值设置较大，一般数据质量佳的情况下推荐设置为5×10^-5到1×10^4之间
    4.分段数据可视化
"""
# 首先，需要确定分段需要用到的markers
# 查看数据中的markers
print(raw.annotations)
# 基于Annotations打印数据的事件持续时长
print(raw.annotations.duration)
# 基于Annotations打印数据的事件的描述信息
print(raw.annotations.description)
# 基于Annotations打印数据的事件的开始时间
print(raw.annotations.onset)


#将Annotations类型的事件信息转为Events类型
events, event_id = mne.events_from_annotations(raw)
# events为记录时间相关的矩阵，event_id为不同markers对应整型的字典信息
# 这里打印出events矩阵的shape和event_id
print(events.shape, event_id)


epochs = mne.Epochs(raw, events, event_id=2, tmin=-1, tmax=2, baseline=(-0.5, 0),
                    preload=True, reject=dict(eeg=2e-4))
# 即分段后的数据存为了Epochs类的对象epochs
# 打印epochs即可看到分段后数据的相关信息
print(epochs)

## 分段数据可视化-（这里显示4个epochs）
epochs.plot(n_epochs=4)
# 绘制功率谱图（逐导联）
epochs.plot_psd(picks='eeg')

# 绘制功率谱拓扑图（分Theta、Alpha和Beta频段）
bands = [(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta')]
epochs.plot_psd_topomap(bands=bands, vlim='joint')











"""
Step 6 叠加平均
     MNE中使用Epochs类来存储分段数据，用Evoked类来存储叠加平均数据
  1.数据叠加平均


"""
# 数据叠加平均
evoked = epochs.average()
# 可视化叠加平均后的数据
# 绘制逐导联的时序信号图
evoked.plot()

# 绘制地形图
# 绘制0ms、0.5s、1s、1.5s和2s处的地形图
times = np.linspace(0, 2, 5)
evoked.plot_topomap(times=times, colorbar=True)

# 绘制某一特定时刻的地形图
# 此例绘制0.8s处，取0.75-0.85s的均值
evoked.plot_topomap(times=0.8, average=0.1)

# 绘制联合图
evoked.plot_joint()

# 绘制逐导联热力图
evoked.plot_image()

# 绘制拓扑时序信号图
evoked.plot_topo()

# 绘制平均所有电极后的ERP
mne.viz.plot_compare_evokeds(evokeds=evoked, combine='mean')

# 绘制枕叶电极的平均ERP
mne.viz.plot_compare_evokeds(evokeds=evoked, picks=['O1', 'Oz', 'O2'], combine='mean')










"""
Step 7 时频分析
      MNE提供了三种时频分析计算方法，分别是：

            Morlet wavelets，对应mne.time_frequency.tfr_morlet()
            DPSS tapers，对应mne.time_frequency.tfr_multitaper()
            Stockwell Transform，对应mne.time_frequency.tfr_stockwell()
            这里，使用第一种方法为例
"""
## 时频分析
# 计算能量（Power）与试次间耦合（inter-trial coherence，ITC）

# 设定一些时频分析的参数
# 频段选取4-30Hz
freqs = np.logspace(*np.log10([4, 30]), num=10)
n_cycles = freqs / 2.
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True)
"""
返回的power即为能量结果，itc即为试次间耦合结果
MNE中时频分析默认返回试次平均后的结果
如果想获取每个试次单独的时频分析结果，将average参数设为False即可
对应代码进行如下修改即可：
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=False)
"""

## 时频结果绘制
"""
MNE的时频绘图方法里可以进行多种baseline矫正方法的选择
其对应参数为mode，包括以下一些选择：

'mean'，减去baseline均值
'ratio'，除以baseline均值
'logratio'，除以baseline均值并取log
'percent'，减去baseline均值并除以baseline均值
'zscore'，减去baseline均值再除以baseline标准差
'zlogratio'，除以baseline均值并取log再除以baseline取log后的标准差
"""
# 绘制结果 枕叶导联的power结果
power.plot(picks=['O1', 'Oz', 'O2'], baseline=(-0.5, 0), mode='logratio', title='auto')
# 绘制枕叶导联的平均power结果
power.plot(picks=['O1', 'Oz', 'O2'], baseline=(-0.5, 0), mode='logratio', title='Occipital', combine='mean')


# 绘制power拓扑图
power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')

# 绘制不同频率的power拓扑图
# 以theta power和alpha power为例
# 取0-0.5s的结果
power.plot_topomap(tmin=0, tmax=0.5, fmin=4, fmax=8,
                   baseline=(-0.5, 0), mode='logratio', title='Theta')
power.plot_topomap(tmin=0, tmax=0.5, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio', title='Alpha')


# 绘制联合图
# 取-0.5s至1.5s的结果
# 并绘制0.5s时10Hz左右的结果和1s时8Hz左右的结果
power.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-0.5, tmax=1.5,
                 timefreqs=[(0.5, 10), (1, 8)])


# ITC结果绘制类似，以拓扑图为例
itc.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average Inter-Trial coherence')







"""
Step 8 提取数据
        在进行相关计算后，往往希望能提取原始数据矩阵、分段数据矩阵、时频结果矩阵等等
        MNE中，Raw类（原始数据类型）、Epochs类（分段后数据类型）和Evocked类（叠加平均后数据类型）
        提供了--get_data()方法
        AverageTFR类（时频分析后数据类型）提供了--.data属性

"""
## get_data()的使用 ---以epochs为例
epochs_array = epochs.get_data()
# 查看获取的数据
print(epochs_array.shape)
print(epochs_array)
"""
即获取了NumPy Array形式的分段数据
其shape为[80, 32, 385]
分别对应80个试次，32个导联和385个时间点

若想获取eog外的导联数据，则可将上述代码改为：

epochs_array = epochs.get_data(picks=['eeg'])
"""


## .data的使用
power_array = power.data
# 查看获取的数据
print(power_array.shape)
print(power_array)
"""
即获取了NumPy Array形式的时频power结果
其shape为[30, 10, 385]
分别对应30个导联，10个频率和385个时间点
"""














