# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 14:46:26 2022

@author: Administrator
"""
"""
opencv:专门做图像处理的库
1.缩放 2.裁剪 3.九宫格 4.人脸识别 5.车牌识别
6.动物识别 7.猪脸识别->>全生命周期管控

"""
# 常见的图片格式
"""
1.bmp:比较老的格式  不常见 无损  基本没有压缩-体积大
2.jpg(jpeg):用最少的磁盘空间，得到较好的图片质量
3.png:无损压缩的位图片格式（首先）
"""

# 常见的图片类型
"""
黑白
彩色
gif 动图 ---一帧一帧的拿出来

"""
# 图片的本质  由像素点组成，就是一个矩阵。每个元素（）像素点
# 都是在[0,255]之间

"""
深入研究黑白图像
# 每个像素点都是在[0,255]之间
1.位图模式（黑白图像）
仅仅只有一位深度的图像  ->（0，1，1，1，0....）
0->  纯黑色
1->  纯白色
2.灰度图像：[0,255]
有8位深度的图像
（0，0，0，0，0，0，0，0） ->2**0=1
（1，1，1，1，1，1，1，1） ->2**8=256
因为python等编程语言都是从0开始计数的
[1,256] -> [0,255]

"""

"""
深入研究彩色图像
三个通道：RGB
R：red -> 红色
G: green ->绿色
B: blue ->蓝色
色彩三原色
一共三个通道，每个通道8个深度
可以表示256**3个颜色
"""



"""
读取一张黑白图片

"""
import cv2
# opencv 所有路径必须是英文--存在中文时，读取时不报错
# 读取结果位None
path = r"C:\Users\Administrator\Desktop\pic1.jpg"
# path:路径 0:代表灰度级 1: 代表彩色图
img = cv2.imread(path,0)

# 读进来的图片是  class 'numpy.ndarray'
print(type(img))
# 读取图片储存形式[h-w]
# 储存类型：uint8 -> 无符号八位整型(0,1,0,1,0,1,1,1)
print(img.shape)

cv2.imshow("imgirl", img)


"""
图片的裁剪

"""
import cv2
# opencv 所有路径必须是英文--存在中文时，读取时不报错
# 读取结果位None
path = r"C:\Users\Administrator\Desktop\pic1.jpg"
# path:路径 0:代表灰度级 1: 代表彩色图
img = cv2.imread(path,0)

# 图片本质： 数组、矩阵
img1 = img[:900,:1200]
cv2.imshow("imgirl", img1)


### numpy 叠加图片
import numpy as np
path = r"C:\Users\Administrator\Desktop\pic1.jpg"
# path:路径 0:代表灰度级 1: 代表彩色图
img = cv2.imread(path,0)

path = r"C:\Users\Administrator\Desktop\pic2.jpg"
# path:路径 0:代表灰度级 1: 代表彩色图
img1 = cv2.imread(path,0)

new_img = img[:800,:300]
new_img1 = img1[:300,:300]

# 垂直拼接 列必须相等
new_img3 = np.vstack((new_img,new_img1))
cv2.imshow("myfavorite", new_img3)

# 水平拼接  行必须相等
new_img = img[:300,:1200]
new_img1 = img1[:300,:400]

new_img4 = np.hstack((new_img,new_img1))
cv2.imshow("myfavoritegirl", new_img4)


"""
图片的保存  
"""
# imwrite(path,target)  可以改变保存的图片的 格式
cv2.imwrite(r"C:\Users\Administrator\Desktop\pic3.jpg", new_img4)

"""
随机的生成图片 
"""
import numpy as np
array1 = np.random.randint(0,255,(700,500,1),dtype="uint8")
cv2.imshow("zhaokai",array1 )



"""
读取一张彩色图片
"""
import cv2
img2 = cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",1)

# 检查彩色图片是否是 array 这种形式
print(type(img2))

# z注意  需要传递图片名称以及图片矩阵
cv2.imshow("zhouye", img2)
# 彩色图片的形状
print(img2.shape)



##### 研究三通道[在opencv里面，以三个二维矩阵进行储存，组成为三位矩阵]

import numpy as np
# 创建一个二维数组
np.array([[1,2],[1,2]]).shape
# 创建一个3维数组
np.array([[[1,2],[1,2]],[[1,2],[1,2]]]).shape

np.array([[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]]]).shape


"""
颜色通道的差分与通道的合并
"""
# 尝试拆分一张彩色图片

## 第一种方法

img3 = cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",1)

# b
img_1 = img3[:,:,0]
cv2.imshow("b", img_1)
cv2.imwrite(r"C:\Users\Administrator\Desktop\pic3_b.jpg", img_1)
# g
img_2 = img3[:,:,1]
cv2.imshow("g", img_2)
cv2.imwrite(r"C:\Users\Administrator\Desktop\pic3_g.jpg", img_2)
# r
img_3 = img3[:,:,2]
cv2.imshow("r", img_3)
cv2.imwrite(r"C:\Users\Administrator\Desktop\pic3_r.jpg", img_3)

## 第二种方法 cv2.split()
b,g,r = cv2.split(img3)


# 合并拆分的结果
img4 = cv2.merge([b,g,r])
cv2.imshow("zhouyea",img4)



"""
利用opencv进行色彩空间转换
1.色彩空间：【bgr】
1）HSV：更类似于人类感觉颜色的方式 H-色相（hub）
        S：饱和度 （saturation） V：亮度(value)
2) YUV: Y:亮度信号 U\V：两个色彩信号  色彩的饱和度
3) Lab:由国际照明委员会建立。L：整张图片的亮度
   a\b：负责颜色的多少        
"""
img3 = cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",1)

cv2.imshow("my girl",img3)

# HSV
HSV = cv2.cvtColor(img3,cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",HSV)

# YUV
YUV= cv2.cvtColor(img3,cv2.COLOR_BGR2YUV)
cv2.imshow("YUV",YUV)

# Lab
Lab= cv2.cvtColor(img3,cv2.COLOR_BGR2Lab)
cv2.imshow("Lab",Lab)

# 有其他的色彩空间   太多了




"""
统计一张图片的像素点大小
"""

# 黑白图片
import numpy as np
import matplotlib.pyplot as plt
import cv2
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic1.jpg",0)

h,w = np.shape(img)
hest = np.zeros([256],dtype=np.int32)
# 遍历图片矩阵
for row in range(h):
    for col in range(w):
        pv = img[row,col]
        hest[pv] +=1
# 绘图
plt.plot(hest,color="r")
plt.xlim([0, 256])
plt.show()






# 彩色图片
import numpy as np
import matplotlib.pyplot as plt
import cv2
img3= cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",1)
# 分离颜色通道
b,g,r = cv2.split(img3)
# 行的遍历
h,w = np.shape(b)
#  建立空白数组
hest = np.zeros([256],dtype=np.int32)
# 遍历图片矩阵
for i in [b,g,r]:
    for row in range(h):
        for col in range(w):
            pv = i[row,col]
            hest[pv] +=1
# 绘图
plt.plot(hest,color="r")
plt.xlim([0, 256])
plt.show()           





# 纯黑纯白像素点统计  -换一下路径就好

"""
随机的生成彩色图片 
"""
import numpy as np
import cv2
array1 = np.random.randint(0,255,(700,500,3),dtype="uint8")
cv2.imshow("zhaokai",array1 )
       
array2 = np.random.randint(200,255,(700,500,3),dtype="uint8")
cv2.imshow("zhaoli",array2 )

new_array = np.hstack((array1,array2))
cv2.imshow("mama",new_array )


"""
彩色图片颜色通道拾遗 
"""
# 彩色图片 ->bgr
import cv2

img3= cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",1)

# 查看b[蓝色]
img_b = img3.copy()
img_b[:,:,1]=0
img_b[:,:,2]=0
cv2.imshow("img_b",img_b)

# 查看g[绿色]
img_g = img3.copy()
img_g[:,:,0]=0
img_g[:,:,2]=0
cv2.imshow("img_g",img_g)

# 查看r[红色]
img_r = img3.copy()
img_r[:,:,0]=0
img_r[:,:,1]=0
cv2.imshow("img_r",img_r)



"""
深入研究图像读取
cv2.imread(path,way)
0:读取灰度图片
1:读取彩色图片[默认]
-1: 读取图片，加载alpha 通道--图片的透明与不透明度
官方公式：Y=0.299R+0.587G+0.114B
    

"""
import cv2

img3= cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",1)

### 以0的方式读取彩色图片
img= cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",0)
cv2.imshow("heibai",img)

# 彩色图像： 三维 BGR ->1799*1200
# 灰度图像：   二维 -> 1799*1200
img3= cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",1)
# 三通道分离
b,g,r = cv2.split(img3)
# new_img = (b+g+r)/3
new_img = 0.114*b+0.587*g+0.299*r
new_img  = new_img.astype("uint8")
cv2.imshow("new_img",new_img)

  
### 以1的方式读取彩灰度图片           
import cv2

img_color = cv2.imread(r"C:\Users\Administrator\Desktop\pic1.jpg",1)
cv2.imshow("new_img",img_color)

"""
官方的填充 ： 以初始的灰度二维矩阵进行RGB通道的填充

"""
img= cv2.imread(r"C:\Users\Administrator\Desktop\pic1.jpg",0)
new_img1 = cv2.merge([img,img,img])
# 一定要检查这个图像的dtype
cv2.imshow("new_img",new_img1)

# 选取部分元素
img_color[100,200]
new_img1[100,200]

"""
灰度图片像素的选取与修改

"""
import cv2

img = cv2.imread(r"C:\Users\Administrator\Desktop\pic1.jpg",0)
cv2.imshow("img",img)

# 选取灰度图像的某个像素点【矩阵-numpy中的数组】

img[100,100]

img[100,200]

# 选取灰度图像的某些像素点【矩阵-numpy中的数组】
img[100:200,100:200]

# 修改灰度图像的某些 像素点【矩阵-numpy中的数组】
img1 = img.copy()
img1[100:600,100:800] = 0
cv2.imshow("img1",img1)

img1[100:600,100:800] = 255
cv2.imshow("img1",img1)

# 0:表示在按任意键关闭  具体数字：表示毫秒
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
彩色图片像素的选取与修改

"""
import cv2

img3 = cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",1)
cv2.imshow("zhouye-my_girl",img3)

# 选取彩色图像的某个像素点【矩阵-numpy中的数组】
# BGR
img3[100,100]

img3[100,100,0]
img3[100,100,1]
img3[100,100,2]


# 选取彩色图像的某些像素点【矩阵-numpy中的数组】
img3[100:200,300:400]

img3[100:200,300:400,0]
img3[100:200,300:400,1]
img3[100:200,300:400,2]

# 修改某个像素点
img3_1 = img3.copy()
img3_1[100,100,0]=0

# 修改某些像素点
img3_1[100:200,300:400,0]=0
img3_1[100:200,300:400,2]=0
cv2.imshow("zy", img3_1)



"""
获取图像的基本属性

"""
# 灰度图片
import cv2

img = cv2.imread(r"C:\Users\Administrator\Desktop\pic1.jpg",0)
cv2.imshow("img",img)

# 形状 [返回一个二维的元组]
print(img.shape)
# 大小 行*列 (像素点的个数)
print(img.size)
1920*1280
# 图像的数据类型 class 'numpy.ndarray'
print(type(img))
# 像素点的数据类型  uint8
print(img.dtype)

# 彩色图片
img3 = cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",1)
cv2.imshow("zhouye-my_girl",img3)

# 形状 [返回一个三维的元组]
print(img3.shape)
# 大小 行*列*通道数 (像素点的个数)
print(img3.size)
1799*1200*3
# 图像的数据类型 class 'numpy.ndarray'
print(type(img3))
# 像素点的数据类型  uint8
print(img3.dtype)



"""
提取感兴趣的ROI区域

"""
import cv2
img3 = cv2.imread(r"C:\Users\Administrator\Desktop\pic3.jpg",1)
cv2.imshow("before",img3)

# 知道图片的shape (1799, 1200, 3)
print(img3.shape)
new_zhouye = img3[200:1000,250:900]
cv2.imshow("after",new_zhouye)

# 写出ROI区域
cv2.imwrite(r"C:\Users\Administrator\Desktop\pic3_1.jpg", new_zhouye)

# 把ROI区域  沾到原始图像上面
img3[:800,:650] = new_zhouye
cv2.imshow("gaixie", img3)

# 把ROI区域  沾到其他图像上面
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",1)

img[:800,:650] = new_zhouye
cv2.imshow("gaixie1", img)


"""
图像的加法

"""
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",1)
cv2.imshow("zya",img4)
# 加法 【前提：必须同样大小的矩阵：对应元素的求和】
img4_1 = img4.copy()

"""
像素点：[0,255]超过255怎么办？【取余】
比如：150+150 = 300
300%255=45  像素点就为45
"""
result1 = img4 + img4_1
cv2.imshow("result1", result1)

"""
opencv  add() 函数
像素点：[0,255]超过255怎么办？【按照255算】
"""
result2 = cv2.add(img4,img4_1)
cv2.imshow("result2", result2)


"""
图像的融合：将两张图片放在一张上
前提：图片的shape 必须一样
0.6*img5 + 0.3*img6 + gamma
"""
import cv2
img5 = cv2.imread(r"C:\Users\Administrator\Desktop\pic5.jfif",1)
img6 = cv2.imread(r"C:\Users\Administrator\Desktop\pic6.jfif",1)

# 图片融合
# addweighted(src1,alpha,src2,beta,gamma)
# src1/src2 第一张图片和第二张图片
# alpha/beta 第一张图片和第二张图片的权重
# gamma： 亮度的调节
result3 = cv2.addWeighted(img5, 0.6, img6, 0.3, 0)
cv2.imshow("result3", result3)

#gamma

result3 = cv2.addWeighted(img5, 0.6, img6, 0.3, 100)
cv2.imshow("result3", result3)

"""
图像的类型转换
转换方式200多种
自己去查想要的转换方式
"""
# 读取了彩色图片
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",1)
cv2.imshow("zya",img4)

# 原始图片BGR  转换为 GRB(本质即为0和2列进行了对调)
# cvtcolor(img,way)
img4_rgb = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
cv2.imshow("img4_rgb", img4_rgb)

# 原始图片BGR  转换为 gray

img4_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
cv2.imshow("img4_gray", img4_gray)

#读取灰度图片

img = cv2.imread(r"C:\Users\Administrator\Desktop\pic1.jpg",1)
cv2.imshow("mn",img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("mn_gray",img_gray)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("mn_rgb",img_rgb)



"""
图像的缩放
本质：矩阵的变换
cv2.resize(src,dsize)
src:原图片 ---需要缩放的图片 img(100*100)
dsize:缩放到的大小(200*200)[列(w)*行(h)]    
"""
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",1)
cv2.imshow("zya",img4)

size = (500,700)
new_img = cv2.resize(img4, size)
cv2.imshow("new_img",new_img)

# 改进--可以按照指定的倍数来缩放
h,w,ch = img4.shape
size = (int(1.1*w),int(0.8*h))
new_img = cv2.resize(img4, size)
cv2.imshow("new_img",new_img)
"""

cv2.resize(src,dsize,fx,fy)
src:原图片 ---需要缩放的图片 img(100*100)
dsize:None  
fx:x方向的比例
fy:y方向的比例
"""
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",1)
cv2.imshow("zya",img4)

new_img2 = cv2.resize(img4,None,fx=1.1,fy=0.8)
cv2.imshow("new_img2",new_img2)



"""
图像的翻转
1.沿着x轴进行翻转
2.沿着y轴进行翻转
3.同时沿着x和y轴进行翻转
cv2.flip(src,flipCode)
src:原图像
flipCode:翻转形式
0：沿着x轴进行翻转
1（大于等于1）：沿着y轴进行翻转
-1（小于等于-1）：同时沿着x和y轴进行翻转

"""
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",1)
cv2.imshow("zya",img4)
# x轴
new_img4 = cv2.flip(img4,0)
cv2.imshow("zya1",new_img4)
# y轴
new_img4 = cv2.flip(img4,1)
cv2.imshow("zya1",new_img4)
# 同时沿着x和y轴进行翻转
new_img4 = cv2.flip(img4,-1)
cv2.imshow("zya1",new_img4)


"""
图像阈值化处理
ret,dst = cv2.threshold(src,thresh,maxval,type)
ret:阈值的返回值（阈值设定的是多少）
dst:输出的图像
src:原图像
thresh:人为指定的阈值
maxval:当像素超过了阈值，小于阈值所赋予的值，否则等于0
type:
    1.cv2.THRESH_BINARY---当像素点大于阈值时，取指定的255；
                          小于阈值的时候，取0
    2.cv2.THRESH_BINARY_INV---取cv2.THRESH_BINARY反                    
出现其他颜色主要是由于 彩色BGR三通道的叠加
     3.cv2.THRESH_TRUNC:超过阈值则取阈值
            低于阈值，则取自身
    4.cv2.THRESH_ TOZERO:超过阈值不变，低于阈值取0
    5.cv2.THRESH_ TOZERO_INV:取反
"""
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",1)

ret,img_2 = cv2.threshold(img4, 127,255,cv2.THRESH_BINARY_INV)

ret,img_3 = cv2.threshold(img4, 127,255,cv2.THRESH_BINARY)
cv2.imshow("zya2",img_2)
cv2.imshow("zya3",img_3)

b,g,r = cv2.split(img_3)
cv2.imshow("b",b)
cv2.imshow("g",g)


# 其他的阈值转换方式
# 1
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",0)
cv2.imshow("zya",img4)
ret,img_4 = cv2.threshold(img4, 127,255,cv2.THRESH_TRUNC)
cv2.imshow("zya4",img_4)

# 2
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",0)

ret,img_4 = cv2.threshold(img4, 127,255,cv2.THRESH_TOZERO)
cv2.imshow("zya4",img_4)

ret,img_5 = cv2.threshold(img4, 127,255,cv2.THRESH_TOZERO_INV)
cv2.imshow("zya5",img_5)
cv2.imshow("zya4",img_4)


"""
图像的  均值滤波---针对缺失像素位置的填补 取均值
cv2.blur(src,kernel)
src:源图像
kernel:大小  （选择多大的矩阵进行移动【3*3最常见】）
"""
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",0)
cv2.imshow("zya",img4)

new_img5 = cv2.blur(img4,(1,1))
cv2.imshow("new_img5", new_img5)

"""
图像的  方框滤波---针对缺失像素位置的填补 
         1.取均值
         2.求和
cv2.boxFilter(src,depth,ksize)
src:源图像
depth:图像的深度，填-1  表示与源图像深度相同
ksize:核大小 (3,3) (5,5)
normalize:是否进行归一化
0:False,不进行归一化，（求和，倾诉点溢出，超过255取255）
1:True，进行归一化，也就是进行均值滤波 
"""
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",0)
cv2.imshow("zya",img4)


new_img6 = cv2.boxFilter(img4,-1,(3,3),normalize = 0)
cv2.imshow("new_img6",new_img6)

new_img6 = cv2.boxFilter(img4,-1,(1,1),normalize = 1)
cv2.imshow("new_img6",new_img6)

"""
图像的  高斯滤波---考虑了像素之间权重的问题
        
cv2.GaussianBlur(src,ksize,sigmaX,sigmaY)
src:源图像
ksize:核大小 (3,3) (5,5) -在高斯滤波中必须是基数
sigmaX,sigmaY:高斯核函数在x和y方向上的标准偏差---控制权重
sigmaX = 0，sigmaX = 0.3*((ksize -1)*0.5-1)+0.8


"""
import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",0)
cv2.imshow("zya",img4)

new_img7 = cv2.GaussianBlur(img4,(1,1),0)
cv2.imshow("new_img7",new_img7)


"""
图像的  中值滤波----取核内的所有数值的中值
        
cv2.medianBlur(src,ksize,sigmaX,sigmaY)
src:源图像
ksize:核大小  只能传递int.(1,3,5,7,9)

"""

import cv2
img4 = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",0)
cv2.imshow("zya",img4)

new_img8 = cv2.medianBlur(img4,3)
cv2.imshow("new_img8",new_img8)


"""
图像的 腐蚀   ---降噪
1.形态学转换主要针对于二值图像（黑白图像（非0即1）
2.卷积核
cv2.reode(src,kernel,iterations)
src:源图像
kernel:卷积核 (3*3)(5*5)(7*7)
iterations: 迭代次数   

"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic2.jpg",0)
cv2.imshow("mn",img)

kernel = np.ones((3,3),np.uint8)
new_img =cv2.erode(img, kernel,iterations = 10)
cv2.imshow("mn1",new_img)



"""
图像的 膨胀  ---降噪

cv2.dilate(src,kernel,iterations)
src:源图像
kernel:卷积核 (3*3)(5*5)(7*7)
iterations: 迭代次数   
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic2.jpg",0)


kernel = np.ones((3,3),np.uint8)
new_img =cv2.dilate(img, kernel,iterations = 10)
cv2.imshow("mn",img)
cv2.imshow("mn1",new_img)


"""
图像的 开运算 --针对于图像外有噪声
腐蚀：去除图片的噪声
膨胀：恢复原来的图片（没有噪声的）
开运算：腐蚀 -> 膨胀
cv2.morphologyEx(src,cv2.MORPH_OPEN,ksize)
src:源图像
cv2.MORPH_OPEN:开运算 （先腐蚀后膨胀这两个步骤）
ksize:卷积核的大小（表示腐蚀及膨胀的大小）
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic2.jpg",0)


kernel = np.ones((7,7),np.uint8)
new_img =cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow("mn",img)
cv2.imshow("mn1",new_img)


"""
图像的 闭运算 --- 针对于图像内有噪声
腐蚀：去除图片的噪声
膨胀：恢复原来的图片（没有噪声的）
闭运算：膨胀-> 腐蚀
cv2.morphologyEx(src,cv2.MORPH_CLOSE,ksize)
src:源图像
cv2.MORPH_CLOSE:闭运算 （先腐蚀后膨胀这两个步骤）
ksize:卷积核的大小（表示腐蚀及膨胀的大小）
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic2.jpg",0)


kernel = np.ones((5,5),np.uint8)
new_img =cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
cv2.imshow("mn",img)
cv2.imshow("mn1",new_img)

# 闭运算更好



"""
图像的梯度运算
图像的梯度：目的在于为了提取目标图像的轮廓
cv2.morphologyEx(src,cv2.MORPH_GRADIENT,ksize)
本质：膨胀-腐蚀
src:源图像
cv2.MORPH_GRADIENT:本质就是 膨胀-腐蚀
ksize:卷积核的大小（表示腐蚀及膨胀的大小）
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic2.jpg",0)

# 彩色图片也可以


kernel = np.ones((3,3),np.uint8)
# 图像的膨胀操作

img1 =cv2.dilate(img, kernel,iterations = 3)



# 图像的腐蚀操作

img2 =cv2.erode(img, kernel,iterations = 3)



# 梯度的计算
new_img = img1 - img2
cv2.imshow("img",img)
cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("new_img",new_img)

# 函数运算
img3 =cv2.morphologyEx(img, cv2.MORPH_GRADIENT,kernel)
cv2.imshow("img3",img3)




"""
图像的礼帽操作--图像外部噪声保留
---目的在于为了提取目标图像的噪声数据
本质：原始图像 - 开运算之后的图像
cv2.morphologyEx(src, cv2.MORPH_TOPHAT,kernel)
src:源图像
cv2.MORPH_TOPHAT:礼貌操作【原始图像-开运算之后的图像】
ksize:卷积核的大小（表示腐蚀及膨胀的大小）
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic2.jpg",0)

kernel = np.ones((9,9),np.uint8)
img4 =cv2.morphologyEx(img, cv2.MORPH_TOPHAT,kernel)
cv2.imshow("img",img)
cv2.imshow("img4",img4)


#### 闭运算   原始图像 - 闭运算之后的图像思路
img5 =cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
new_img = img5 - img
cv2.imshow("img",img)
cv2.imshow("img5",img5)
cv2.imshow("new_img",new_img)
#  思路不可行

"""
图像的黑帽操作---图像内部噪声保留
---目的在于为了提取目标图像的噪声数据
本质：闭运算-原始图像 之后的图像
cv2.morphologyEx(src, cv2.MORPH_BLACKHAT,kernel)
src:源图像
cv2.MORPH_BLACKHAT:黑帽操作【闭运算-原始图像 】
ksize:卷积核的大小（表示腐蚀及膨胀的大小）
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic2.jpg",0)

img6 =cv2.morphologyEx(img, cv2.MORPH_BLACKHAT,kernel)
cv2.imshow("img",img)
cv2.imshow("img6",img6)





"""
sobel 算子 ：图像边缘检测的一种方式---本质是一种梯度运算
cv2.Sobel(src,depth,dx,dy,ksize)
src:源图像
depth:图像的深度【-1表示与源图像相同 --表示剪断，0-255】
[cv2.CV_64F:保留负数部分]
dx\dy：x和y方向的梯度
ksize:卷积核的大小（表示腐蚀及膨胀的大小）
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic2.jpg",0)

# 计算x方向的梯度
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize = 3)

# 将负数转换为整数
sobelx = cv2.convertScaleAbs(sobelx)

cv2.imshow("sobelx", sobelx)

# 计算y方向的梯度
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize = 3)

# 将负数转换为整数
sobely = cv2.convertScaleAbs(sobely)

cv2.imshow("sobely", sobely)

# addweighted()  完成图像融合
new_img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv2.imshow("new_img",new_img)


"""
直接计算融合的x和y的梯度
同时计算x和y的梯度
建议：不要直接同时计算x和y
"""
sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize = 3)
sobelxy = cv2.convertScaleAbs(sobelxy)

cv2.imshow("sobelxy",sobelxy)













"""
Scharr 算子 ：图像边缘检测的一种方式
Sobel 算子的改进,能够发现不明显的轮廓
本质就是卷积核的不一样。增加了核心点像素附近的权重
cv2.Scharr(src,depth,dx,dy)
src:源图像
depth:图像的深度【-1表示与源图像相同 】
dx\dy：x和y方向的梯度
 1.dx\dy均大于等于0
 2.dx + dy ==1
 3.只能先计算x y ,然后再进行融合
ksize:卷积核的大小默认 3*3  无需设置
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic2.jpg",0)

# 计算x方向的梯度
scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
scharrx = cv2.convertScaleAbs(scharrx)
cv2.imshow("scharrx",scharrx)



# 计算y方向的梯度
scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
scharry = cv2.convertScaleAbs(scharry)
cv2.imshow("scharry",scharry)

# addweighted()  完成图像融合
new_img = cv2.addWeighted(scharrx, 0.5,scharry, 0.5, 0)
cv2.imshow("new_img",new_img)



"""
Laplacian(拉普拉斯) 算子 ：图像边缘检测的一种方式
对噪声比较敏感  一般不单独使用
cv2.Laplacian(src,depth)
src:源图像
depth:图像的深度【-1表示与源图像相同 --表示剪断，0-255】
[cv2.CV_64F:保留负数部分]
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic2.jpg",0)

new_img = cv2.Laplacian(img,cv2.CV_64F)
new_img= cv2.convertScaleAbs(new_img)
cv2.imshow("img",img)
cv2.imshow("new_img",new_img)









"""
Canny 边缘检测
1. 应用高斯滤波器，平滑图像，滤除噪音  降噪
2.计算图像中每个像素点的梯度大小和方向  梯度
3.使用极大值抑制，消除边缘检测带来的不利影响
4.应用双阈值检测  确定真实和潜在的边缘
5.通过抑制孤立的弱边缘完成边缘检测
cv2.Canny(src,min,max)
src:源图像
min:最小阈值
max：最大阈值

"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic7.jfif",0)

new_img = cv2.Canny(img, 80, 200)
cv2.imshow("img",img)
cv2.imshow("new_img",new_img)





"""
图像金字塔的向下采样
cv2.pyrDown(src)
src:源图像
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic7.jfif",0)

t1 = cv2.pyrDown(img)
t2 = cv2.pyrDown(t1)
cv2.imshow("img",img)
cv2.imshow("t1",t1)
cv2.imshow("t2",t2)

"""
图像金字塔的向上采样
cv2.pyrUp(src)
src:源图像
"""
cv2.imwrite(r"C:\Users\Administrator\Desktop\pic8.jpg", t2)

img = cv2.imread(r"C:\Users\Administrator\Desktop\pic8.jpg",0)

t1 = cv2.pyrUp(img)
t2 = cv2.pyrUp(t1)
cv2.imshow("img",img)
cv2.imshow("t1",t1)
cv2.imshow("t2",t2)

"""
拉普拉斯金字塔
"""
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic6.jfif",0)

# 拉普拉斯金字塔的第0层
down = cv2.pyrDown(img)
down_up = cv2.pyrUp(down)
new_img = img - down_up
cv2.imshow("new_img",new_img)

# 拉普拉斯金字塔的第1层
down1 = cv2.pyrDown(down)
down_up1 = cv2.pyrUp(down1)
new_img1 =down - down_up1
cv2.imshow("new_img1",new_img1)


# 拉普拉斯金字塔的第2层
down2 = cv2.pyrDown(down1)
down_up2 = cv2.pyrUp(down2)
new_img2 =down1 - down_up2
cv2.imshow("new_img2",new_img2)


"""
轮廓检测
二值图像
黑色背景，白色对象
cv2.findContours(src,mode,method)
src:源图像
mode : 轮廓检测方式
  cv2.RETR_EXTERNAL: 只检测外轮廓
  cv2.RETR_LIST：  检测的轮廓不建立等级关系
  cv2.RETR_CCOMP:  检测所有的轮廓，并组织为两层
                顶层是各部分的外部边界
                第二层是空洞的边界
  cv2.RETR_TREE:检测所有的轮廓，并重构嵌套轮廓的在整个层次
             
method: 轮廓逼近的方法
    1.cv2.CHAIN_APPROX_NONE:以Freeman链码的方式输出轮廓，
    所有其他的方法输出多边形（顶点的序列） -占内存大
    2.cv2.CHAIN_APPROX_SIMPLE: 压缩水平的、垂直的和斜的部分
    也就是，函数只保留他们的终点部分-占内存小
返回值
 1.image : 修改后的原始图像
 2.contours: 轮廓
 3.hierarchy: 图像的拓扑信息（轮廓层次）

"""

"""
轮廓绘制
cv2.drawContours(src,contours,num,color,width)
src: 源图像
contours: 轮廓信息
num: 绘制多少个轮廓 -1为全部轮廓
color:轮廓绘制的颜色 （0，0，255）BGR 红色
width: 轮廓的宽度   
"""

import cv2
# 读取彩色图片
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",1)

# 将彩色图片转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 将灰度图片转换为二值图
ret,binary = cv2.threshold(gray, 127, 255,cv2.THRESH_BINARY)


# 寻找图像轮廓
contours,hie = \
cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# 复制图像
draw_img = img.copy()

# 绘制图像

new_img = cv2.drawContours(draw_img,contours,-1,(0,255,0),2)


cv2.imshow("img",img)
cv2.imshow("new_img",new_img)




"""
直方图：统计一张图片中每种像素点的个数
图片 -矩阵- 元素- [0，255]
0：100
1：500
"""
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",0)
"""
cv2.calcHist(images, channels, mask, histSize, ranges)
images:源图像-但是需要中括号包裹[]
channels:图像的通道 灰度【0】 彩色[0] [1] [2]-- BGR
mask：掩码 --统计图像的一部分像素点 都统计None
histsize:绘制直方图的个数 256
ranges:绘制像素点的范围 【0，255】
"""

hist = cv2.calcHist([img], [0],None , [256], [0,255])

# 绘制图像
plt.plot(hist, color = "r")
plt.show()

####  彩色图
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",1)


b = cv2.calcHist([img], [0],None , [256], [0,255])
g = cv2.calcHist([img], [1],None , [256], [0,255])
r = cv2.calcHist([img], [2],None , [256], [0,255])

# 绘制图像
plt.plot(b, color = "b")
plt.plot(g, color = "g")
plt.plot(r, color = "r")
plt.show()



"""
图像的掩码：mask
类似于图像的剪裁【但 略有不同】
"""
# 生成掩码
import numpy as np
img = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",0)

mask = np.zeros(img.shape,np.uint8)

# 选取哪一部分进行展示
mask[300:,200:650] = 255

cv2.imshow("mask", mask)

# 与操作
hist_full = cv2.bitwise_and(img,mask)
cv2.imshow("hist_full", hist_full )

# 绘制图像
hist = cv2.calcHist([hist_full], [0],mask , [256], [0,255])
plt.plot(hist, color = "b")
plt.show()

"""
直方图  进行了图像像素点个数的统计   
目的：图像的均衡化处理 --本质-平衡图像矩阵内的元素

 
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",0)

hist = cv2.calcHist([img], [0],None, [256], [0,255])

plt.plot(hist, color = "r")
plt.show()

# 图像的均衡化处理

equ = cv2.equalizeHist(img)
hist1 = cv2.calcHist([equ], [0],None, [256], [0,255])

plt.plot(hist1, color = "g")
cv2.imshow("img",img)
cv2.imshow("equ",equ)

"""
自适应直方图均衡化
解决细节丢失
cv2.createCLAHE(clipLimit = 40,tileGridSize = (6,6)) 
clipLimit:小于cliplimit的像素点不参与均分，大于参与
越接近0表示越接近原图
tileGridSize ：划分多少块

"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Administrator\Desktop\pic4.jpg",0)
equ = cv2.equalizeHist(img)
cv2.imshow("equ",equ)

# 自适应均衡化
model =  cv2.createCLAHE(clipLimit = 40,tileGridSize = (3,3)) 

new_img = model.apply(img)
cv2.imshow("new_img",new_img)








