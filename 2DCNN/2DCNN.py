# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:36:38 2022

@author: dell
这是使用2DVGG进行卷积
"""
import pandas as pd
import time
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from numpy import *
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True                     # 不全部占满显存, 按需分配
config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 限制GPU内存占用率
sess = tf.compat.v1.Session(config=config)
start_time=time.time()d

#1
#读取数据
data1_1=pd.read_csv('./data/Training area dataset/block 1.csv')['G']
data1_2=pd.read_csv('./data/Training area dataset/block 1.csv')['M']
data1_3=pd.read_csv('./data/Training area dataset/block 1.csv')['R']
data1_4=pd.read_csv('./data/Training area dataset/block 1.csv')['L']
x1_data=pd.concat([data1_1,data1_2,data1_3], axis=1)
y1_data=pd.concat([data1_4], axis=1)
x1_data1=np.array(x1_data,dtype=np.float32)
y1_data1=np.array(y1_data,dtype=np.float32)

I=0
J=0
K=0
i=0
j=0
k=0
X_data=[]
Y_data=[]
#定义滑动窗口的尺寸和滑动范围（11*11*11的窗口在276*54*25的分块体里滑动）54*276*25   K，J，i三个循环分别控制滑动窗口向三个方向进行滑动
while K<17:     #滑动窗口沿k方向滑动,步幅为1   25-10
    J=0
    while J<268:    #滑动窗口沿j方向滑动,步幅为1  276-10
        i=0
        while i<46:   #54-10 不同block的水平向东滑动的不同起始位置，如i=0，则0:11，i=1,则1：12，沿着i，步幅为1进行滑动
                j=0
                while j<9:   #按水平向东再向北方向，读取水平方向11*11的数据
                    X_data.append(x1_data1[i+(J+j)*54+(K)*54*276:(i+(J+j)*54+(K)*54*276)+9,:])   #+11  抽象理解：这里i、J、K就是滑动窗口11*11*11左下角体素的索引
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
#取中心点做标签
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:  #25-10
    J=0
    while J<268: #276-10
        i=0
        while i<46:  #54-10
                Y_data.append(y1_data1[i+4+(J+4)*54+K*54*276])   #取大小为9*9*9块体的中心体素的label当为总的label I、J、K方向的中心值
                i=i+1
        J=J+1
    K=K+1

#2
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
while K<17:   #28*236*25
    J=0
    while J<228:
        i=0
        while i<20:
                j=0
                while j<9:
                    X_data.append(x2_data1[i+(J+j)*28+(K)*28*236:(i+(J+j)*28+(K)*28*236)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<228:
        i=0
        while i<20:
            Y_data.append(y2_data1[i+4+(J+4)*28+(K)*28*236])
            i=i+1
        J=J+1
    K=K+1
    
#3
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
while K<17:   #29*26*25
    J=0
    while J<18:
        i=0
        while i<21:
                j=0
                while j<9:
                    X_data.append(x3_data1[i+(J+j)*29+(K)*26*29:(i+(J+j)*29+(K)*26*29)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<18:
        i=0
        while i<21:
            Y_data.append(y3_data1[i+4+(J+4)*29+(K)*26*29])
            i=i+1
        J=J+1
    K=K+1

#4      
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
while K<17:  #276*75*25
    J=0
    while J<67:
        i=0
        while i<268:
                j=0
                while j<9:
                    X_data.append(x4_data1[i+(J+j)*276+(K)*276*75:(i+(J+j)*276+(K)*276*75)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<67:
        i=0
        while i<268:
            Y_data.append(y4_data1[i+4+(J+4)*276+(K)*276*75])
            i=i+1
        J=J+1
    K=K+1

#5    
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
while K<17:  #150*150*25
    J=0
    while J<142:
        i=0
        while i<142:
                j=0
                while j<9:
                    X_data.append(x5_data1[i+(J+j)*150+(K)*150*150:(i+(J+j)*150+(K)*150*150)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<142:
        i=0
        while i<142:
            Y_data.append(y5_data1[i+4+(J+4)*150+(K)*150*150])
            i=i+1
        J=J+1
    K=K+1

#6    
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
while K<17:  #100*100*25
    J=0
    while J<92:
        i=0
        while i<92:
                j=0
                while j<9:
                    X_data.append(x6_data1[i+(J+j)*100+(K)*10000:(i+(J+j)*100+(K)*10000)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<92:
        i=0
        while i<92:
            Y_data.append(y6_data1[4+i+(J+4)*100+(K)*10000])
            i=i+1
        J=J+1
    K=K+1
#7
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
while K<17:  #100*100*25
    J=0
    while J<92:
        i=0
        while i<92:
                j=0
                while j<9:
                    X_data.append(x7_data1[i+(J+j)*100+(K)*10000:(i+(J+j)*100+(K)*10000)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<92:
        i=0
        while i<92:
            Y_data.append(y7_data1[4+i+(J+4)*100+(K)*10000])
            i=i+1
        J=J+1
    K=K+1
 

#8    
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
while K<17:  #60*60*25
    J=0
    while J<52:
        i=0
        while i<52:
                j=0
                while j<9:
                    X_data.append(x8_data1[i+(J+j)*60+(K)*3600:(i+(J+j)*60+(K)*3600)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<52:
        i=0
        while i<52:
            Y_data.append(y8_data1[4+i+(J+4)*60+(K)*3600])
            i=i+1
        J=J+1
    K=K+1

#9 
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
while K<17:  #50*50*25
    J=0
    while J<42:
        i=0
        while i<42:
                j=0
                while j<9:
                    X_data.append(x9_data1[i+(J+j)*50+(K)*2500:(i+(J+j)*50+(K)*2500)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<42:
        i=0
        while i<42:
            Y_data.append(y9_data1[4+i+(J+4)*50+(K)*2500])
            i=i+1
        J=J+1
    K=K+1

#10    
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
while K<17:  #50*50*25
    J=0
    while J<42:
        i=0
        while i<42:
                j=0
                while j<9:
                    X_data.append(x10_data1[i+(J+j)*50+(K)*2500:(i+(J+j)*50+(K)*2500)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<42:
        i=0
        while i<42:
            Y_data.append(y10_data1[4+i+(J+4)*50+(K)*2500])
            i=i+1
        J=J+1
    K=K+1

#11    
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
while K<17:  #85*185*25
    J=0
    while J<177:
        i=0
        while i<77:
                j=0
                while j<9:
                    X_data.append(x11_data1[i+(J+j)*85+(K)*85*185:(i+(J+j)*85+(K)*85*185)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<177:
        i=0
        while i<77:
            Y_data.append(y11_data1[4+i+(J+4)*85+(K)*85*185])
            i=i+1
        J=J+1
    K=K+1

#12    
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
while K<17:  #60*60*25
    J=0
    while J<52:
        i=0
        while i<52:
                j=0
                while j<9:
                    X_data.append(x12_data1[i+(J+j)*60+(K)*3600:(i+(J+j)*60+(K)*3600)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<52:
        i=0
        while i<52:
            Y_data.append(y12_data1[4+i+(J+4)*60+(K)*3600])
            i=i+1
        J=J+1
    K=K+1
    
#13    
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
while K<17:  #75*75*25
    J=0
    while J<67:
        i=0
        while i<67:
                j=0
                while j<9:
                    X_data.append(x13_data1[i+(J+j)*75+(K)*75*75:(i+(J+j)*75+(K)*75*75)+9,:])
                    j=j+1
                i=i+1
        J=J+1
    K=K+1
I=0
J=0
K=0
i=0
j=0
k=0
while K<17:
    J=0
    while J<67:
        i=0
        while i<67:
            Y_data.append(y13_data1[4+i+(J+4)*75+(K)*75*75])
            i=i+1
        J=J+1
    K=K+1
# 格式转换，定义块体个数（-1自动计算）、通道数、长、宽、高
X_data=np.array(X_data)
np.save("./numpy_data/9_9_X.npy",X_data)
print("X data has been saved")
X_data=X_data.reshape(-1,3)
X_data=X_data.reshape(-1,3,9,9)
Y_data=np.array(Y_data)
np.save("./numpy_data/9_9Y.npy",Y_data)
print("Y data has been saved")
Y_data=Y_data.reshape(-1,1)
# 分割训练集与验证集
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size =0.25, random_state=30)
print("There are {} training samples".format(y_train.shape[0]))
print("There are {} testing samples".format(y_test.shape[0]))
#对 标签进行one-hot编码
y_train = np_utils.to_categorical(y_train, num_classes=15)
y_test = np_utils.to_categorical(y_test, num_classes=15)
Y_data = np_utils.to_categorical(Y_data, num_classes=15)
print("数据导入加载及预处理的总时间为{:.2f}".format(time.time()-start_time))
#训 练网络结构设计  #卷积改为Convolution2D
model = Sequential()
##1:64

model.add(Convolution2D(
        64,
        kernel_size=3,
		padding = 'same',
		input_shape=(3,9,9)
						)
		)
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D(
					pool_size = (2,2),
					strides = (2,2),
					padding = 'same',
					)
		)

##2:128
model.add(Convolution2D(128, kernel_size=3,padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2, padding = 'same'))

##3:256
model.add(Convolution2D(256, kernel_size=3,padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2, padding = 'same'))

##4:512
model.add(Convolution2D(512,  kernel_size=3, padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2,padding = 'same'))

##5:512
model.add(Convolution2D(512, kernel_size=3, padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2, padding = 'same'))

#####FC
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))
########################
#优化器
adam = Adam(lr = 0.0002)

########################
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])



print('Training ------------')

summary=model.summary()

#开始训练（train）
train_history=model.fit(X_train, y_train, epochs=25, batch_size=64,validation_data=(X_test, y_test),shuffle=True)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
testloss, testaccuracy = model.evaluate(X_test, y_test) #计算准确率和损失值
trainloss,trainaccuracy= model.evaluate(X_train, y_train)
print('\ntrain loss:{}，train accuracy:{} '.format(trainloss,trainaccuracy))
print('\ntest loss:{}，test accuracy:{}'.format(testloss,testaccuracy))


model.save('./model/9_9vgg2d05.h5')   # HDF5文件，pip install h5py，保存模型



#绘制准确率曲线函数
def show_acc_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='lower right')
    plt.savefig('./model/paper/9_9vgg2d05ACC.png',dpi=300)
    plt.show()
show_acc_history(train_history,'accuracy','val_accuracy')

#绘制loss曲线函数
def show_loss_history(train_history,train,validation):
    plt.plot(train_history.history[train])#训练数据的执行结果
    plt.plot(train_history.history[validation])#验证数据的执行结果
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper right')
    plt.savefig('./model/paper/9_9vgg2d05LOSS.png',dpi=300)
    plt.show()
show_loss_history(train_history,'loss','val_loss')

#绘制混淆矩阵函数
def plot_confusion_matrix(cm, classes,
    title='Confusion matrix',
    cmap=plt.cm.Greens):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./model/paper/9_9vgg2d05MATRIX.png',dpi=300)
    plt.show()

# 显示混淆矩阵函数
def plot_confuse(model, x_val, y_val):
    predictions = model.predict(x_val)
    predictions = np.argmax(predictions,axis=1)
    # predictions = model.predict_classes(x_val)
    #predictions=np.argmax(predictions,axis=-1)
    truelabel = y_val.argmax(axis=-1) # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure(figsize=(10,10),dpi=300)
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))
    #plot_confusion_matrix(conf_mat, range(7))


#绘制评估指标
train_history = load_model('./model/9_9vgg2d05.h5')
plot_confuse(train_history,X_test, y_test)
print("\n2DVGGmodel train is over")