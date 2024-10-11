# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:22:37 2022

@author: dell
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                                # 设置 CUDA 设备的顺序为 PCI 总线 ID
os.environ["CUDA_VISIBLE_DEVICES"] = "1"                                      # 设置可见的 CUDA 设备为第 1 个（从 0 开始计数）
config = tf.compat.v1.ConfigProto()                                           # 创建 TensorFlow 配置对象,用于配置 TensorFlow 运行时的一些选项
config.gpu_options.allow_growth = True                                        # 设置 GPU 选项，允许动态分配显存，以便按需使用
config.gpu_options.per_process_gpu_memory_fraction = 0.99                     # 限制 GPU 内存占用率为 99%
sess = tf.compat.v1.Session(config=config)                                    # 创建 TensorFlow 会话（Session）对象，使用上述配置

X_data = []
Y_data = []
save_file="./data/Prediction area dataset/DataFor3d/9_9_9vgg3dResult(13).csv"
Frames=[]
#
#predict1
csv1_file="./data/Prediction area dataset/DataFor3d/predict1.csv"
data1_1 = pd.read_csv(csv1_file)['G']
data1_2 = pd.read_csv(csv1_file)['M']
data1_3 = pd.read_csv(csv1_file)['R']
x1_data = pd.concat([data1_1, data1_2, data1_3], axis=1)
x1_data1 =np.array(x1_data,dtype=np.float32)
# predict1:125*112*85
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:
    J = 0
    while J < 104:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x1_data1[i+(J+j)*125+(K+k)*125*112:(i+(J+j)*125+(K+k)*125*112)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1

#predict2
csv2_file="./data/Prediction area dataset/DataFor3d/predict2.csv"
data2_1 = pd.read_csv(csv2_file)['G']
data2_2 = pd.read_csv(csv2_file)['M']
data2_3 = pd.read_csv(csv2_file)['R']
x2_data = pd.concat([data2_1, data2_2, data2_3], axis=1)
x2_data1 =np.array(x2_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x2_data1[i+(J+j)*125+(K+k)*125*125:(i+(J+j)*125+(K+k)*125*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1



#predict3
csv3_file="./data/Prediction area dataset/DataFor3d/predict3.csv"
data3_1 = pd.read_csv(csv3_file)['G']
data3_2 = pd.read_csv(csv3_file)['M']
data3_3 = pd.read_csv(csv3_file)['R']
x3_data = pd.concat([data3_1, data3_2, data3_3], axis=1)
x3_data1 =np.array(x3_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x3_data1[i+(J+j)*125+(K+k)*125*125:(i+(J+j)*125+(K+k)*125*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1


# #predict4
csv4_file="./data/Prediction area dataset/DataFor3d/predict4.csv"
data4_1 = pd.read_csv(csv4_file)['G']
data4_2 = pd.read_csv(csv4_file)['M']
data4_3 = pd.read_csv(csv4_file)['R']
x4_data = pd.concat([data4_1, data4_2, data4_3], axis=1)
x4_data1 =np.array(x4_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x4_data1[i+(J+j)*125+(K+k)*125*125:(i+(J+j)*125+(K+k)*125*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1


#predict5
csv5_file="./data/Prediction area dataset/DataFor3d/predict5.csv"
data5_1 = pd.read_csv(csv5_file)['G']
data5_2 = pd.read_csv(csv5_file)['M']
data5_3 = pd.read_csv(csv5_file)['R']
x5_data = pd.concat([data5_1, data5_2, data5_3], axis=1)
x5_data1 =np.array(x5_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x5_data1[i+(J+j)*125+(K+k)*125*125:(i+(J+j)*125+(K+k)*125*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1


#predict6
csv6_file="./data/Prediction area dataset/DataFor3d/predict6.csv"
data6_1 = pd.read_csv(csv6_file)['G']
data6_2 = pd.read_csv(csv6_file)['M']
data6_3 = pd.read_csv(csv6_file)['R']
x6_data = pd.concat([data6_1, data6_2, data6_3], axis=1)
x6_data1 =np.array(x6_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x6_data1[i+(J+j)*125+(K+k)*125*125:(i+(J+j)*125+(K+k)*125*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1
#

#predict7
csv7_file="./data/Prediction area dataset/DataFor3d/predict7.csv"
data7_1 = pd.read_csv(csv7_file)['G']
data7_2 = pd.read_csv(csv7_file)['M']
data7_3 = pd.read_csv(csv7_file)['R']
x7_data = pd.concat([data7_1, data7_2, data7_3], axis=1)
x7_data1 =np.array(x7_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x7_data1[i+(J+j)*125+(K+k)*125*125:(i+(J+j)*125+(K+k)*125*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1

# #predict8
csv8_file="./data/Prediction area dataset/DataFor3d/predict8.csv"
data8_1 = pd.read_csv(csv8_file)['G']
data8_2 = pd.read_csv(csv8_file)['M']
data8_3 = pd.read_csv(csv8_file)['R']
x8_data = pd.concat([data8_1, data8_2, data8_3], axis=1)
x8_data1 =np.array(x8_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*115*85
    J = 0
    while J < 107:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x8_data1[i+(J+j)*125+(K+k)*125*115:(i+(J+j)*125+(K+k)*115*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1
#

#predict9
csv9_file="./data/Prediction area dataset/DataFor3d/predict9.csv"
data9_1 = pd.read_csv(csv9_file)['G']
data9_2 = pd.read_csv(csv9_file)['M']
data9_3 = pd.read_csv(csv9_file)['R']
x9_data = pd.concat([data9_1, data9_2, data9_3], axis=1)
x9_data1 =np.array(x9_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*120*85
    J = 0
    while J < 112:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x9_data1[i+(J+j)*125+(K+k)*125*120:(i+(J+j)*125+(K+k)*120*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1


#predict10
csv10_file="./data/Prediction area dataset/DataFor3d/predict10.csv"
data10_1 = pd.read_csv(csv10_file)['G']
data10_2 = pd.read_csv(csv10_file)['M']
data10_3 = pd.read_csv(csv10_file)['R']
x10_data = pd.concat([data10_1, data10_2, data10_3], axis=1)
x10_data1 =np.array(x10_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x10_data1[i+(J+j)*125+(K+k)*125*125:(i+(J+j)*125+(K+k)*125*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1


#predict11
csv11_file="./data/Prediction area dataset/DataFor3d/predict11.csv"
data11_1 = pd.read_csv(csv11_file)['G']
data11_2 = pd.read_csv(csv11_file)['M']
data11_3 = pd.read_csv(csv11_file)['R']
x11_data = pd.concat([data11_1, data11_2, data11_3], axis=1)
x11_data1 =np.array(x11_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*121*85
    J = 0
    while J < 113:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x11_data1[i+(J+j)*125+(K+k)*125*121:(i+(J+j)*125+(K+k)*121*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1


#predict12
csv12_file="./data/Prediction area dataset/DataFor3d/predict12.csv"
data12_1 = pd.read_csv(csv12_file)['G']
data12_2 = pd.read_csv(csv12_file)['M']
data12_3 = pd.read_csv(csv12_file)['R']
x12_data = pd.concat([data12_1, data12_2, data12_3], axis=1)
x12_data1 =np.array(x12_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x12_data1[i+(J+j)*125+(K+k)*125*125:(i+(J+j)*125+(K+k)*125*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1


#predict13
csv13_file="./data/Prediction area dataset/DataFor3d/predict13.csv"
data13_1 = pd.read_csv(csv13_file)['G']
data13_2 = pd.read_csv(csv13_file)['M']
data13_3 = pd.read_csv(csv13_file)['R']
x13_data = pd.concat([data13_1, data13_2, data13_3], axis=1)
x13_data1 =np.array(x13_data,dtype=np.float32)
I=0
J=0
K=0
i=0
j=0
k=0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            k = 0
            while k < 9:
                j = 0
                while j < 9:
                    X_data.append(x13_data1[i+(J+j)*125+(K+k)*125*125:(i+(J+j)*125+(K+k)*125*125)+9,:])
                    j=j+1
                k=k+1
            i = i + 1
        J = J + 1
    K = K + 1


x_data = np.array(X_data)
x_data = x_data.reshape(-1, 3)
x_data = x_data.reshape(-1, 3, 9,9,9)                       # 其中 -1 表示自动计算，3 表示第一个维度的长度，后面的三个维度表示每个子数组的形状

model_file="./model/3DCNNWithoutDrop/3DCNN.h5"
model = load_model(model_file)

character = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']  # 定义标签类别
print('Predict-------------------')
predict = model.predict(x_data)                             # 进行模型预测
predict = np.argmax(predict, axis=1)                        # 获取模型预测结果中概率最大的类别索引
tmp = predict[0]                                            # 获取第一个样本的预测类别索引
char = character[tmp]                                       # 根据类别索引获取对应的字符或标签
conclusion = np.array(predict, dtype=np.int8)               # 将所有样本的预测类别索引转换为整数类型的数组
print(conclusion)
prediction_label = np.array(conclusion, dtype=np.int8)      # 创建预测标签数组，数据类型为整数

save = pd.DataFrame(prediction_label)                       # 将预测标签数组转换为 DataFrame 对象
print(save)
print('Export---------------')

# 将预测结果对应到各待估点上
#prediction1
data1_X = pd.read_csv(csv1_file)['X']
data1_Y = pd.read_csv(csv1_file)['Y']
data1_Z = pd.read_csv(csv1_file)['Z']
data1_ID = pd.read_csv(csv1_file)['ID']
frames1 = pd.concat([data1_ID, data1_X, data1_Y, data1_Z], axis=1)
frames1_1 = np.array(frames1, dtype=np.float32)
Frames = []
K = 0
J = 0
i = 0
while K < 77:  #125*112*85
    J = 0
    while J < 104:
        i = 0
        while i < 117:
            Frames.append(frames1_1[i+4+(J+4)*125+(K+4)*125*112, :])
            i = i + 1
        J = J + 1
    K = K + 1


#prediction2
data2_X = pd.read_csv(csv2_file)['X']
data2_Y = pd.read_csv(csv2_file)['Y']
data2_Z = pd.read_csv(csv2_file)['Z']
data2_ID = pd.read_csv(csv2_file)['ID']
frames2 = pd.concat([data2_ID, data2_X, data2_Y, data2_Z], axis=1)
frames2_1 = np.array(frames2, dtype=np.float32)
K = 0
J = 0
i = 0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            Frames.append(frames2_1[i+4+(J+4)*125+(K+4)*125*125, :])
            i = i + 1
        J = J + 1
    K = K + 1


#prediction3
data3_X = pd.read_csv(csv3_file)['X']
data3_Y = pd.read_csv(csv3_file)['Y']
data3_Z = pd.read_csv(csv3_file)['Z']
data3_ID = pd.read_csv(csv3_file)['ID']
frames3 = pd.concat([data3_ID, data3_X, data3_Y, data3_Z], axis=1)
frames3_1 = np.array(frames3, dtype=np.float32)
K = 0
J = 0
i = 0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            Frames.append(frames3_1[i+4+(J+4)*125+(K+4)*125*125, :])
            i = i + 1
        J = J + 1
    K = K + 1


# #prediction4
data4_X = pd.read_csv(csv4_file)['X']
data4_Y = pd.read_csv(csv4_file)['Y']
data4_Z = pd.read_csv(csv4_file)['Z']
data4_ID = pd.read_csv(csv4_file)['ID']
frames4 = pd.concat([data4_ID, data4_X, data4_Y, data4_Z], axis=1)
frames4_1 = np.array(frames4, dtype=np.float32)
K = 0
J = 0
i = 0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            Frames.append(frames4_1[i+4+(J+4)*125+(K+4)*125*125, :])
            i = i + 1
        J = J + 1
    K = K + 1


#prediction5
data5_X = pd.read_csv(csv5_file)['X']
data5_Y = pd.read_csv(csv5_file)['Y']
data5_Z = pd.read_csv(csv5_file)['Z']
data5_ID = pd.read_csv(csv5_file)['ID']
frames5 = pd.concat([data5_ID, data5_X, data5_Y, data5_Z], axis=1)  
frames5_1 = np.array(frames5, dtype=np.float32)   
K = 0
J = 0
i = 0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            Frames.append(frames5_1[i+4+(J+4)*125+(K+4)*125*125, :])  
            i = i + 1
        J = J + 1
    K = K + 1


#prediction6
data6_X = pd.read_csv(csv6_file)['X']
data6_Y = pd.read_csv(csv6_file)['Y']
data6_Z = pd.read_csv(csv6_file)['Z']
data6_ID = pd.read_csv(csv6_file)['ID']
frames6 = pd.concat([data6_ID, data6_X, data6_Y, data6_Z], axis=1)  
frames6_1 = np.array(frames6, dtype=np.float32)   
K = 0
J = 0
i = 0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            Frames.append(frames6_1[i+4+(J+4)*125+(K+4)*125*125, :])  
            i = i + 1
        J = J + 1
    K = K + 1


#prediction7
data7_X = pd.read_csv(csv7_file)['X']
data7_Y = pd.read_csv(csv7_file)['Y']
data7_Z = pd.read_csv(csv7_file)['Z']
data7_ID = pd.read_csv(csv7_file)['ID']
frames7 = pd.concat([data7_ID, data7_X, data7_Y, data7_Z], axis=1)  
frames7_1 = np.array(frames7, dtype=np.float32)   
K = 0
J = 0
i = 0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            Frames.append(frames7_1[i+4+(J+4)*125+(K+4)*125*125, :])  
            i = i + 1
        J = J + 1
    K = K + 1
#

#prediction8
data8_X = pd.read_csv(csv8_file)['X']
data8_Y = pd.read_csv(csv8_file)['Y']
data8_Z = pd.read_csv(csv8_file)['Z']
data8_ID = pd.read_csv(csv8_file)['ID']
frames8 = pd.concat([data8_ID, data8_X, data8_Y, data8_Z], axis=1)  
frames8_1 = np.array(frames8, dtype=np.float32)   
K = 0
J = 0
i = 0
while K < 77:  #125*115*85
    J = 0
    while J < 107:
        i = 0
        while i < 117:
            Frames.append(frames8_1[i+4+(J+4)*125+(K+4)*125*115, :])  
            i = i + 1
        J = J + 1
    K = K + 1

#prediction9
data9_X = pd.read_csv(csv9_file)['X']
data9_Y = pd.read_csv(csv9_file)['Y']
data9_Z = pd.read_csv(csv9_file)['Z']
data9_ID = pd.read_csv(csv9_file)['ID']
frames9 = pd.concat([data9_ID, data9_X, data9_Y, data9_Z], axis=1)  
frames9_1 = np.array(frames9, dtype=np.float32)   
K = 0
J = 0
i = 0
while K < 77:  #125*120*85
    J = 0
    while J < 112:
        i = 0
        while i < 117:
            Frames.append(frames9_1[i+4+(J+4)*125+(K+4)*125*120, :])  
            i = i + 1
        J = J + 1
    K = K + 1

#prediction10
data10_X = pd.read_csv(csv10_file)['X']
data10_Y = pd.read_csv(csv10_file)['Y']
data10_Z = pd.read_csv(csv10_file)['Z']
data10_ID = pd.read_csv(csv10_file)['ID']
frames10 = pd.concat([data10_ID, data10_X, data10_Y, data10_Z], axis=1)  
frames10_1 = np.array(frames10, dtype=np.float32)   
K = 0
J = 0
i = 0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            Frames.append(frames10_1[i+4+(J+4)*125+(K+4)*125*125, :])  
            i = i + 1
        J = J + 1
    K = K + 1

#prediction11
data11_X = pd.read_csv(csv11_file)['X']
data11_Y = pd.read_csv(csv11_file)['Y']
data11_Z = pd.read_csv(csv11_file)['Z']
data11_ID = pd.read_csv(csv11_file)['ID']
frames11 = pd.concat([data11_ID, data11_X, data11_Y, data11_Z], axis=1)
frames11_1 = np.array(frames11, dtype=np.float32)
K = 0
J = 0
i = 0
while K < 77:  #125*121*85
    J = 0
    while J < 113:
        i = 0
        while i < 117:
            Frames.append(frames11_1[i+4+(J+4)*125+(K+4)*125*121, :])
            i = i + 1
        J = J + 1
    K = K + 1

#prediction12
data12_X = pd.read_csv(csv12_file)['X']
data12_Y = pd.read_csv(csv12_file)['Y']
data12_Z = pd.read_csv(csv12_file)['Z']
data12_ID = pd.read_csv(csv12_file)['ID']
frames12 = pd.concat([data12_ID, data12_X, data12_Y, data12_Z], axis=1)
frames12_1 = np.array(frames12, dtype=np.float32)
K = 0
J = 0
i = 0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            Frames.append(frames12_1[i+4+(J+4)*125+(K+4)*125*125, :])
            i = i + 1
        J = J + 1
    K = K + 1

#prediction13
data13_X = pd.read_csv(csv13_file)['X']
data13_Y = pd.read_csv(csv13_file)['Y']
data13_Z = pd.read_csv(csv13_file)['Z']
data13_ID = pd.read_csv(csv13_file)['ID']
frames13 = pd.concat([data13_ID, data13_X, data13_Y, data13_Z], axis=1)
frames13_1 = np.array(frames13, dtype=np.float32)
K = 0
J = 0
i = 0
while K < 77:  #125*125*85
    J = 0
    while J < 117:
        i = 0
        while i < 117:
            Frames.append(frames13_1[i+4+(J+4)*125+(K+4)*125*125, :])
            i = i + 1
        J = J + 1
    K = K + 1

Frames = pd.DataFrame(Frames)
framescat = pd.concat([Frames, save], ignore_index=True, axis=1)
# 保存结果
save1 = pd.DataFrame(framescat)
print(save1)
save1.to_csv(save_file,index=False, header=False)

print('-------Over-------')
