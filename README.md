# 基于卷积神经网络的深部地层岩性预测

岩石地层建模在矿产资源勘探和地质研究中起着至关重要的作用。在这项研究中，我们介绍了一种利用反转地球物理特性在深地下自动化伪岩石建模的新方法。我们提出了一种具有自适应矩估计 （3D Adam-CNN） 的三维卷积神经网络来实现这一目标。我们的模型采用 3D 地球物理特性作为训练的输入特征，同时重建浅层地下的 3D 地质模型以进行岩石学标记。
模型基于Tensorflow2.70框架，参照经典的VGG框架设计了用于深部地层岩性预测的网络框架，以下为实验用到的数据、代码及环境说明。  

论文链接：[Deep Subsurface Pseudo-Lithostratigraphic Modeling Based on Three-Dimensional Convolutional Neural Network (3D CNN) Using Inversed Geophysical Properties and Shallow Subsurface Geological Model](https://pubs.geoscienceworld.org/gsw/lithosphere/article/2024/1/lithosphere_2023_273/634861/Deep-Subsurface-Pseudo-Lithostratigraphic-Modeling "论文")

***

# 目录  
- [文件介绍](#文件介绍)  
- [安装](#安装)  
- [鸣谢](#鸣谢)  
  
# 文件介绍  
文件列表包括以下内容：
- README
- installment
- dataset 
- 1DCNN
- 2DCNN
- 3DCNN
- Distribution histogram

**README**：项目及文件介绍

**installment**：训练代码所使用的环境依赖信息，方便环境配置，以requirement.txt和environment.yml不同格式的文件保存。 

**dataset**:实验的数据集，包括训练数据集和预测数据集。

**1DCNN**:采用一维滑动窗口获取数据并使用一维卷积核进行卷积操作的实验代码，包括1DCNN（1DCNN训练实验代码）、1DPrediciton(1DCNN预测实验代码)。

**2DCNN**：采用二维滑动窗口获取数据并使用二维卷积核进行卷积操作的实验代码，包括2DCNN（2DCNN训练实验代码）、2DPrediciton(2DCNN预测实验代码)。

**3DCNN**：采用三维滑动窗口获取数据并使用三维卷积核进行卷积操作的实验代码，包括3DCNN（3DCNN训练实验代码）、3DPrediciton(3DCNN预测实验代码)。
**Distribution histogram**:用于绘制论文附图中的拟合曲线及柱状分布图的代码。

# 安装  
1. 使用environment.yml创建环境

```
conda env create -f environment.yml
```
解读：在base环境下执行上述指令，会直接创建一个新的环境，并在该环境下，安装相应依赖项
2. 使用requirements.txt创建环境  
```
pip install -r requirements.txt
```  
解读：在当前环境下安装相应依赖项，如果需要在其他环境下安装依赖项，可以先创建并激活新环境，再使用上述命令。 
  
# 鸣谢  
本研究得到了国家自然科学基金（批准号：42072326）和中国地质调查局工作项目（批准号：DD20190156）的资助。
