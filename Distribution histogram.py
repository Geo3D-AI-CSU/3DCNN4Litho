import numpy as np
import matplotlib.pyplot as plt


# 根据均值、标准差,求指定范围的正态分布概率值
def normfun(x, mu, sigma):
  pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
  return pdf

# 生成随机数据，使用正态分布
result = np.random.normal(15, 44, 100)                # 均值为15, 方差为44的正态分布，生成100个随机数
print(result)

x = np.arange(min(result), max(result), 0.1)          # 生成 x 轴的数据，范围从生成的随机数的最小值到最大值，步长为0.1
# 计算理论的正态分布概率密度函数值
print(result.mean(), result.std())
y = normfun(x, result.mean(), result.std())
plt.plot(x, y)

# 画出实际的参数概率与取值关系，生成直方图
plt.hist(result, bins=10, rwidth=0.8, density=True)   # 10个柱状图，宽度是rwidth(0~1)，density=True表示归一化
plt.title('distribution')
plt.xlabel('temperature')
plt.ylabel('probability')
# 输出
plt.show()                                            # 最后图片的概率和不为1是因为正态分布是从负无穷到正无穷,这里指截取了数据最小值到最大值的分布