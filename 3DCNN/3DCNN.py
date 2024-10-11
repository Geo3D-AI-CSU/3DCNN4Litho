# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:36:38 2022

@author: dell
"""
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from numpy import *
from tensorflow.keras.layers import Convolution3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"                              # 设置 CUDA 设备的顺序为 PCI 总线 ID
os.environ["CUDA_VISIBLE_DEVICES"]="1"                                    # 设置可见的 CUDA 设备为第 1 个（从 0 开始计数）
config = tf.compat.v1.ConfigProto()                                       # 创建 TensorFlow 配置对象,用于配置 TensorFlow 运行时的一些选项。
config.gpu_options.allow_growth = True                                    # 设置 GPU 选项，允许动态分配显存，以便按需使用
config.gpu_options.per_process_gpu_memory_fraction = 0.99                 # 限制 GPU 内存占用率为 99%
sess = tf.compat.v1.Session(config=config)                                # 创建 TensorFlow 会话（Session）对象，使用上述配置

# block1
# 读取数据
data1_1=pd.read_csv('./data/Training area dataset/block 1.csv')['G']      # 密度对比
data1_2=pd.read_csv('./data/Training area dataset/block 1.csv')['M']      # 磁化率
data1_3=pd.read_csv('./data/Training area dataset/block 1.csv')['R']      # 视电阻率
data1_4=pd.read_csv('./data/Training area dataset/block 1.csv')['L']      # 岩性类别标签
x1_data=pd.concat([data1_1,data1_2,data1_3], axis=1)                      # 沿着列方向连接成一个新的 DataFrame
y1_data=pd.concat([data1_4], axis=1)                                      # 取第四列即label类形成y
x1_data1=np.array(x1_data,dtype=np.float32)                               # 将dataframe转成浮点型Numpy数组
y1_data1=np.array(y1_data,dtype=np.float32)                               # 将dataframe转成浮点型Numpy数组

I=0
J=0
K=0
i=0
j=0
k=0
X_data=[]                                                                # 定义空列表存储给定滑动窗口重采样的X数据（三个地球物理参数）
Y_data=[]                                                                # 定义空列表存储给定滑动窗口重采样的Y数据（岩性类别标签）
# 定义滑动窗口的尺寸和滑动范围（11*11*11的窗口在54*276*25的分块体里滑动）
# 前面三个（K、J、i）循环控制了滑动窗口的最左下角的格网索引，后面两个循环（k，j）定义了k、j方向的滑动窗口大小
while K<15:                                                             # 滑动窗口最左下角格网的k方向索引
    J=0
    while J<266:                                                        # 滑动窗口最左下角格网的j方向索引
        i=0
        while i<44:                                                     # 滑动窗口最左下角格网的i方向索引
            k=0
            while k<11:                                                 # k方向滑动窗口的大小
                j=0
                while j<11:                                             # j方向滑动窗口的大小
                    X_data.append(x1_data1[i+(J+j)*54+(K+k)*54*276:(i+(J+j)*54+(K+k)*54*276)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
# 取中心点做标签
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:                                                           # 滑动窗口最左下角的gird的Z方向的索引，从0开始，25-10=15
    J=0
    while J<266:                                                      # 滑动窗口最左下角的gird的y方向的索引，从0开始，276-10=266
        i=0
        while i<44:                                                   # 滑动窗口最左下角的gird的X方向的索引，从0开始，54-10=44
            Y_data.append(y1_data1[i+5+(J+5)*54+(K+5)*54*276])        # 取滑动窗口几何中心的gird标签当整个block的标签0-10中心是5
            i=i+1
        J=J+1
    K=K+1

# block2
data2_1=pd.read_csv('./data/Training area dataset/block 2.csv')['G']
data2_2=pd.read_csv('./data/Training area dataset/block 2.csv')['M']
data2_3=pd.read_csv('./data/Training area dataset/block 2.csv')['R']
data2_4=pd.read_csv('./data/Training area dataset/block 2.csv')['L']
x2_data=pd.concat([data2_1,data2_2,data2_3], axis=1)
y2_data=pd.concat([data2_4], axis=1)
x2_data1=np.array(x2_data,dtype=np.float32)
y2_data1=np.array(y2_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<226:
        i=0
        while i<18:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x2_data1[i+(J+j)*28+(K+k)*28*236:(i+(J+j)*28+(K+k)*28*236)+11,:])
               
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<226:
        i=0
        while i<18:
            Y_data.append(y2_data1[i+5+(J+5)*28+(K+5)*28*236])
            #print(i+3+J*46+K*13248+2*46+2*13248)
            i=i+1
        J=J+1
    K=K+1
    
# block3
data3_1=pd.read_csv('./data/Training area dataset/block 3.csv')['G']
data3_2=pd.read_csv('./data/Training area dataset/block 3.csv')['M']
data3_3=pd.read_csv('./data/Training area dataset/block 3.csv')['R']
data3_4=pd.read_csv('./data/Training area dataset/block 3.csv')['L']
x3_data=pd.concat([data3_1,data3_2,data3_3], axis=1)
y3_data=pd.concat([data3_4], axis=1)
x3_data1=np.array(x3_data,dtype=np.float32)
y3_data1=np.array(y3_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<16:
        i=0
        while i<19:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x3_data1[i+(J+j)*29+(K+k)*26*29:(i+(J+j)*29+(K+k)*26*29)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<16:
        i=0
        while i<19:
            Y_data.append(y3_data1[i+5+(J+5)*29+(K+5)*26*29])
            i=i+1
        J=J+1
    K=K+1

# block4
data4_1=pd.read_csv('./data/Training area dataset/block 4.csv')['G']
data4_2=pd.read_csv('./data/Training area dataset/block 4.csv')['M']
data4_3=pd.read_csv('./data/Training area dataset/block 4.csv')['R']
data4_4=pd.read_csv('./data/Training area dataset/block 4.csv')['L']
x4_data=pd.concat([data4_1,data4_2,data4_3], axis=1)
y4_data=pd.concat([data4_4], axis=1)
x4_data1=np.array(x4_data,dtype=np.float32)#
y4_data1=np.array(y4_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<65:
        i=0
        while i<266:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x4_data1[i+(J+j)*276+(K+k)*276*75:(i+(J+j)*276+(K+k)*276*75)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<65:
        i=0
        while i<266:
            Y_data.append(y4_data1[i+5+(J+5)*276+(K+5)*276*75])
            i=i+1
        J=J+1
    K=K+1

# block5
data5_1=pd.read_csv('./data/Training area dataset/block 5.csv')['G']
data5_2=pd.read_csv('./data/Training area dataset/block 5.csv')['M']
data5_3=pd.read_csv('./data/Training area dataset/block 5.csv')['R']
data5_4=pd.read_csv('./data/Training area dataset/block 5.csv')['L']
x5_data=pd.concat([data5_1,data5_2,data5_3], axis=1)
y5_data=pd.concat([data5_4], axis=1)
x5_data1=np.array(x5_data,dtype=np.float32)
y5_data1=np.array(y5_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<140:
        i=0
        while i<140:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x5_data1[i+(J+j)*150+(K+k)*150*150:(i+(J+j)*150+(K+k)*150*150)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<140:
        i=0
        while i<140:
            Y_data.append(y5_data1[i+5+(J+5)*150+(K+5)*150*150])
            i=i+1
        J=J+1
    K=K+1

# block6
data6_1=pd.read_csv('./data/Training area dataset/block 6.csv')['G']
data6_2=pd.read_csv('./data/Training area dataset/block 6.csv')['M']
data6_3=pd.read_csv('./data/Training area dataset/block 6.csv')['R']
data6_4=pd.read_csv('./data/Training area dataset/block 6.csv')['L']
x6_data=pd.concat([data6_1,data6_2,data6_3], axis=1)
y6_data=pd.concat([data6_4], axis=1)
x6_data1=np.array(x6_data,dtype=np.float32)#
y6_data1=np.array(y6_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<90:
        i=0
        while i<90:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x6_data1[i+(J+j)*100+(K+k)*10000:(i+(J+j)*100+(K+k)*10000)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<90:
        i=0
        while i<90:
            Y_data.append(y6_data1[5+i+(J+5)*100+(K+5)*10000])
            i=i+1
        J=J+1
    K=K+1
# block7
data7_1=pd.read_csv('./data/Training area dataset/block 7.csv')['G']
data7_2=pd.read_csv('./data/Training area dataset/block 7.csv')['M']
data7_3=pd.read_csv('./data/Training area dataset/block 7.csv')['R']
data7_4=pd.read_csv('./data/Training area dataset/block 7.csv')['L']
x7_data=pd.concat([data7_1,data7_2,data7_3], axis=1)
y7_data=pd.concat([data7_4], axis=1)
x7_data1=np.array(x7_data,dtype=np.float32)
y7_data1=np.array(y7_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<90:
        i=0
        while i<90:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x7_data1[i+(J+j)*100+(K+k)*10000:(i+(J+j)*100+(K+k)*10000)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<90:
        i=0
        while i<90:
            Y_data.append(y7_data1[5+i+(J+5)*100+(K+5)*10000])
            i=i+1
        J=J+1
    K=K+1
 

# block8
data8_1=pd.read_csv('./data/Training area dataset/block 8.csv')['G']
data8_2=pd.read_csv('./data/Training area dataset/block 8.csv')['M']
data8_3=pd.read_csv('./data/Training area dataset/block 8.csv')['R']
data8_4=pd.read_csv('./data/Training area dataset/block 8.csv')['L']
x8_data=pd.concat([data8_1,data8_2,data8_3], axis=1)
y8_data=pd.concat([data8_4], axis=1)
x8_data1=np.array(x8_data,dtype=np.float32)#
y8_data1=np.array(y8_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<50:
        i=0
        while i<50:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x8_data1[i+(J+j)*60+(K+k)*3600:(i+(J+j)*60+(K+k)*3600)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<50:
        i=0
        while i<50:
            Y_data.append(y8_data1[5+i+(J+5)*60+(K+5)*3600])
            i=i+1
        J=J+1
    K=K+1

# block9
data9_1=pd.read_csv('./data/Training area dataset/block 9.csv')['G']
data9_2=pd.read_csv('./data/Training area dataset/block 9.csv')['M']
data9_3=pd.read_csv('./data/Training area dataset/block 9.csv')['R']
data9_4=pd.read_csv('./data/Training area dataset/block 9.csv')['L']
x9_data=pd.concat([data9_1,data9_2,data9_3], axis=1)
y9_data=pd.concat([data9_4], axis=1)
x9_data1=np.array(x9_data,dtype=np.float32)#
y9_data1=np.array(y9_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<40:
        i=0
        while i<40:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x9_data1[i+(J+j)*50+(K+k)*2500:(i+(J+j)*50+(K+k)*2500)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<40:
        i=0
        while i<40:
            Y_data.append(y9_data1[5+i+(J+5)*50+(K+5)*2500])
            i=i+1
        J=J+1
    K=K+1

# block10
data10_1=pd.read_csv('./data/Training area dataset/block 10.csv')['G']
data10_2=pd.read_csv('./data/Training area dataset/block 10.csv')['M']
data10_3=pd.read_csv('./data/Training area dataset/block 10.csv')['R']
data10_4=pd.read_csv('./data/Training area dataset/block 10.csv')['L']
x10_data=pd.concat([data10_1,data10_2,data10_3], axis=1)
y10_data=pd.concat([data10_4], axis=1)
x10_data1=np.array(x10_data,dtype=np.float32)#
y10_data1=np.array(y10_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<40:
        i=0
        while i<40:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x10_data1[i+(J+j)*50+(K+k)*2500:(i+(J+j)*50+(K+k)*2500)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<40:
        i=0
        while i<40:
            Y_data.append(y10_data1[5+i+(J+5)*50+(K+5)*2500])
            i=i+1
        J=J+1
    K=K+1

# block11
data11_1=pd.read_csv('./data/Training area dataset/block 11.csv')['G']
data11_2=pd.read_csv('./data/Training area dataset/block 11.csv')['M']
data11_3=pd.read_csv('./data/Training area dataset/block 11.csv')['R']
data11_4=pd.read_csv('./data/Training area dataset/block 11.csv')['L']
x11_data=pd.concat([data11_1,data11_2,data11_3], axis=1)
y11_data=pd.concat([data11_4], axis=1)
x11_data1=np.array(x11_data,dtype=np.float32)#
y11_data1=np.array(y11_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<5:
    J=0
    while J<175:
        i=0
        while i<75:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x11_data1[i+(J+j)*85+(K+k)*85*185:(i+(J+j)*85+(K+k)*85*185)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<5:
    J=0
    while J<175:
        i=0
        while i<75:
            Y_data.append(y11_data1[5+i+(J+5)*85+(K+5)*85*185])
            i=i+1
        J=J+1
    K=K+1

# block12
data12_1=pd.read_csv('./data/Training area dataset/block 12.csv')['G']
data12_2=pd.read_csv('./data/Training area dataset/block 12.csv')['M']
data12_3=pd.read_csv('./data/Training area dataset/block 12.csv')['R']
data12_4=pd.read_csv('./data/Training area dataset/block 12.csv')['L']
x12_data=pd.concat([data12_1,data12_2,data12_3], axis=1)
y12_data=pd.concat([data12_4], axis=1)
x12_data1=np.array(x12_data,dtype=np.float32)#
y12_data1=np.array(y12_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<50:
        i=0
        while i<50:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x12_data1[i+(J+j)*60+(K+k)*3600:(i+(J+j)*60+(K+k)*3600)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<50:
        i=0
        while i<50:
            Y_data.append(y12_data1[5+i+(J+5)*60+(K+5)*3600])
            i=i+1
        J=J+1
    K=K+1
    
# block13
data13_1=pd.read_csv('./data/Training area dataset/block 13.csv')['G']
data13_2=pd.read_csv('./data/Training area dataset/block 13.csv')['M']
data13_3=pd.read_csv('./data/Training area dataset/block 13.csv')['R']
data13_4=pd.read_csv('./data/Training area dataset/block 13.csv')['L']
x13_data=pd.concat([data13_1,data13_2,data13_3], axis=1)
y13_data=pd.concat([data13_4], axis=1)
x13_data1=np.array(x13_data,dtype=np.float32)#
y13_data1=np.array(y13_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<65:
        i=0
        while i<65:
            k=0
            while k<11:
                j=0
                while j<11:
                    X_data.append(x13_data1[i+(J+j)*75+(K+k)*75*75:(i+(J+j)*75+(K+k)*75*75)+11,:])
                    j=j+1
                k=k+1
            i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<15:
    J=0
    while J<65:
        i=0
        while i<65:
            Y_data.append(y13_data1[5+i+(J+5)*75+(K+5)*75*75])
            i=i+1
        J=J+1
    K=K+1
# 格式转换，定义块体个数（-1自动计算）、通道数、长、宽、高
X_data=np.array(X_data)
X_data=X_data.reshape(-1,3)
X_data=X_data.reshape(-1,3,11,11,11)

Y_data=np.array(Y_data)
Y_data=Y_data.reshape(-1,1)

# 分割训练集与验证集
# train_test_split函数来自sklearn.model_selection模块。它用于将数据集分割成训练集和测试集，以便在机器学习模型中进行训练和评估
# 使用 train_test_split 函数从 X_data 和 Y_data 中分割出训练集（X_train, y_train）和测试集（X_test, y_test）

# 参数说明：
# - X_data: 特征数据
# - Y_data: 目标数据
# - test_size: 测试集占总数据的比例，这里设置为 0.25，即 25% 的数据用于测试
# - random_state: 随机种子，用于保证每次运行程序时得到的分割结果是一致的

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size =0.25, random_state=30)

#对标签进行one-hot编码
# 使用 np_utils.to_categorical 将类别标签进行 one-hot 编码
# np_utils.to_categorical 函数来自 Keras 的 np_utils 模块

# 参数说明：
# - y_train: 训练集标签
# - y_test: 测试集标签
# - Y_data: 总体数据集标签
# - num_classes: 类别的总数，这里设置为 15，表示有 15 个类别

# 在深度学习中，通常会将类别标签转换成 one-hot 编码的形式，以便更好地与模型进行训练和预测。

y_train = np_utils.to_categorical(y_train, num_classes=15)
y_test = np_utils.to_categorical(y_test, num_classes=15)
Y_data = np_utils.to_categorical(Y_data, num_classes=15)


# 训练网络结构设计
model = Sequential()                                                   # 使用 Sequential 模型，它是 Keras 中的一种简单线性堆叠模型
## Module1:Conv-64
# 添加一个 3D 卷积层（Convolutional Layer）:
# - 64: 输出的特征图数目
# - kernel_dim1=3, kernel_dim2=3, kernel_dim3=3: 卷积核的维度（深度、行、列）
# - border_mode='same': 采用 "same" 边界模式，保持输入和输出的尺寸相同
# - input_shape=(3, 11, 11, 11): 输入的形状，表示输入是一个 3D 的数据，深度为 3，行、列均为 11
model.add(Convolution3D(
        64,
        kernel_dim1=3, # depth
        kernel_dim2=3, # rows
        kernel_dim3=3, # cols
		border_mode = 'same',
		input_shape=(3,11,11,11)
						)
		)
model.add(BatchNormalization())                                        # 添加批标准化层，有助于加速训练过程并提高模型的稳定性
model.add(Activation('relu'))                                          # 添加激活函数层，这里使用 ReLU 激活函数
model.add(Dropout(0.2))                                                # 添加一个 Dropout 层，用于防止过拟合，随机丢弃 20% 的神经元
model.add(MaxPooling3D(
					pool_size = (2,2,2),
					strides = (2,2,2),
					border_mode = 'same',
					)
		)

## Module2:Conv-128
model.add(Convolution3D(128, 3,3,3,border_mode = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling3D(2, 2, border_mode = 'same'))

## Module3:Conv-256
model.add(Convolution3D(256,3, 3,3,border_mode = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling3D(2, 2, border_mode = 'same'))

## Module4:Conv-512
model.add(Convolution3D(512, 3, 3,3, border_mode = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling3D(2, 2,border_mode = 'same'))

## Module5:Conv-512
model.add(Convolution3D(512, 3,3,3, border_mode = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling3D(2, 2, border_mode = 'same'))

## Module6:FC-512
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))

## Optimizer
adam = Adam(lr = 0.0002)



# 编译深度学习模型
# # 参数说明：
# # - optimizer='adam': 优化器选择 Adam，它是一种基于梯度下降的优化算法，通常在深度学习中表现良好
# # - loss='categorical_crossentropy': 损失函数选择交叉熵损失，用于多类别分类问题，其中输出是 one-hot 编码的标签
# # - metrics=['accuracy']: 评估指标选择准确率，用于衡量模型在训练和测试过程中的性能

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')

summary=model.summary()                                        # 使用 model.summary() 方法，它打印出模型的架构和参数统计信息

#开始训练（train）
# 参数说明：
# - X_train: 训练集的特征数据
# - y_train: 训练集的标签数据
# - epochs=100: 训练轮数，模型将对整个训练集进行多次迭代学习，这里设置为 100 轮
# - batch_size=64: 每个训练批次中包含的样本数，用于梯度下降的批量更新，这里设置为 64
# - validation_data=(X_test, y_test): 验证集的特征和标签数据，模型将在每个训练轮后在验证集上进行评估
# - shuffle=True: 是否在每个训练轮之前打乱训练集的顺序，有助于更好地学习模型

train_history=model.fit(X_train, y_train, epochs=100, batch_size=64,validation_data=(X_test, y_test),shuffle=True)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)                 # 计算准确率和损失值

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

# HDF5文件，pip install h5py，保存模型
model.save('I:/qingniandian/train area of qingniandian/test/based point/DATA/lithology_csv20220318/model/Lithology20220505_model_9m1_vgg_Wei.h5')

# 绘制accuracy曲线函数
# 参数说明：
# - train_history: 训练历史，包含了训练过程中的各种指标的变化情况
# - train: 训练集准确率的指标名称，例如 'accuracy'
# - validation: 验证集准确率的指标名称，例如 'val_accuracy'

def show_acc_history(train_history,train,validation):
    # 绘制训练集和验证集准确率的变化曲线
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    # 添加图表标签
    plt.ylabel(train)
    plt.xlabel('epoch')
    # 添加图例
    plt.legend(['train','validation'],loc='lower right')
    # 保存图表为图片文件
    plt.savefig('I:/qingniandian/train area of qingniandian/test/based point/DATA/lithology_csv20220318/model/paper/acc 9m1.png')
    # 显示图表
    plt.show()

show_acc_history(train_history,'accuracy','val_accuracy')       # 调用函数显示训练历史中准确率的变化曲线图

# 绘制loss曲线函数
# 参数说明：
# - train_history: 训练历史，包含了训练过程中的各种指标的变化情况
# - train: 训练集损失函数的指标名称，例如 'loss'
# - validation: 验证集损失函数的指标名称，例如 'val_loss'

def show_loss_history(train_history,train,validation):
    # 绘制训练集和验证集损失函数的变化曲线
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    # 添加轴标签
    plt.xlabel('epoch')
    # 添加图例
    plt.legend(['train','validation'],loc='upper right')
    # 保存图表为图片文件
    plt.savefig('I:/qingniandian/train area of qingniandian/test/based point/DATA/lithology_csv20220318/model/paper/loss 9m1.png')
    # 显示图表
    plt.show()

show_loss_history(train_history,'loss','val_loss')               # 调用函数显示训练历史中损失函数的变化曲线图

# 绘制混淆矩阵函数
# 参数说明：
# - cm: 混淆矩阵
# - classes: 类别标签
# - title: 图表标题，默认为 'Confusion matrix'
# - cmap: 颜色映射，默认使用 Greens

def plot_confusion_matrix(cm, classes,
    title='Confusion matrix',
    cmap=plt.cm.Greens):
    # 对混淆矩阵进行归一化处理，以便更好地显示分类准确率
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 绘制归一化后的混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # 添加标题
    plt.title(title)
    # 添加颜色条
    plt.colorbar()
    # 设置坐标轴刻度及标签
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # 在图表中添加标注
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    # 调整布局
    plt.tight_layout()
    # 设置坐标轴标签
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # 保存图表为图片文件
    plt.savefig('I:/qingniandian/train area of qingniandian/test/based point/DATA/lithology_csv20220318/model/paper/Confusion matrix 9m1.png')
    # 显示图表
    plt.show()

# 计算混淆矩阵函数
# 参数说明：
# - model: 已经训练好的深度学习模型
# - x_val: 验证集的特征数据
# - y_val: 验证集的标签数据

def plot_confuse(model, x_val, y_val):
    # 使用模型进行预测
    predictions = model.predict_classes(x_val)
    # 获取真实标签：沿着最后一个轴（通常是类别轴）找到每行中的最大值的索引
    truelabel = y_val.argmax(axis=-1)
    # 计算混淆矩阵
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    # 绘制混淆矩阵图
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))


# # 载入模型并显示混淆矩阵
train_history = load_model('I:/qingniandian/train area of qingniandian/test/based point/DATA/lithology_csv20220318/model/Lithology20220505_model_9m1_vgg_Wei.h5')
plot_confuse(train_history,X_test, y_test)