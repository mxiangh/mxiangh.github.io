---
title: 机器学习（4）核回归——kernel Regression
tags: ML Regression
---

#### 核回归——Kernel Regression

思想：核回归是一种非参数回归方法，用于估计变量间的非线性关系，通过局部加权平均来拟合数据，权重由核函数决定。与线性回归不同，核回归不需要预设全局模型形式，而是通过数据驱动的方式灵活拟合局部特征。

<!--more-->

##### 1.原理说明

核回归类似一句古话“近朱者赤，近墨者黑”，我们认为一个样本点会受到其周边的样本点的影响，例如，如果和好学生坐的近成绩可能会更好一些，和成绩差的学生坐一起成绩可能会低一些。

远近关系体现了距离，而距离的影响体现的是权重。

1.1 权重

为了刻画这种权重，引入核函数来描述距离的影响，这里以高斯核（类似高斯分布）为例：

$$ K_h(x)=e^{-\frac{(x-x_i)^2}{2 h^{2}}} $$

$x_i$是其他的样本，x是待预测样本。

可以发现，如果$x_i$和x距离越大，它们的差值越大，k反而越小，也就是较远的点分配的权重反而小。

如果$x_i$和x距离越小，它们的差值越小，k反而越大，也就是较近的点分配的权重反而大。

公式里还有一个h，被称为带宽。带宽h决定了这种 “局部影响” 的范围，h越大，更多远的样本点会被赋予相对大的权重，回归曲线会更平滑；h越小，只有非常靠近x的样本点影响大，曲线会更灵活但易受噪声影响 。

1.2 分配权重

因为要预测y值，则要考虑所有样本点$y_i$的取值，且让距离预测点近的样本点的$y_i$在占比更高，突出局部样本对预测值的贡献

$$ \sum_{i=1}^{n} K_{h}\left(x-x_{i}\right) y_{i} $$

1.3 归一化

如果只是加权求和，得出来的值其实并不合理，因为不同特征可能有不同量纲（如身高用 “厘米”、体重用 “千克” ），数值范围差异大，所以需要进行归一化处理，让结果更合理。

##### 2.预测值计算公式：

$$\widehat{m}\left(x_{0}\right)=\widehat{E}\left[Y \mid X=x_{0}\right]=\frac{\sum_{i=1}^{n} K_{h}\left(x-x_{i}\right) y_{i}}{\sum_{i=1}^{n} K_{h}\left(x-x_{i}\right)}$$

- x：待预测的点；
- $x_i，y_i$：已知观测点的数据；
- K（·）：核函数。

##### 3.常见核函数：

- 高斯核：$ K_h(u)=e^{-\frac{u^{2}}{2 h^{2}}} （常用）$

- Epanechnikov核：$ K_h(u)=max(0,1-\frac{u^{2}}{h^{2}})$

- 均匀核：
  
  $$ K_h(u)=\left\{\begin{array}{l}
  1, \text { if } \vert u \vert \le h \\
  0, \text { otherwise }
  \end{array}\right.$$

- 三角核：$ K_h(u)=max(0,1-\frac{\vert u \vert}{h})$

##### 4.算法流程：

（1）选择核函数与带宽

（2）循环预测点$x$

- 计算$x$与样本点$x_i$的距离；
  
- 将距离代入核函数中计算每个样本点的权重；
  
- 计算加权权重和；
  
- 计算预测值m($x$)

##### 5.举个例子：

假设我们有以下观测数据（温度x，销量y）：

| 温度°C | 销量 |
| ------ | ---- |
| 10     | 50   |
| 15     | 70   |
| 20     | 100  |
| 25     | 120  |
| 30     | 150  |

目标：预测当温度为18°C时的销量。

步骤一：选择高斯核和带宽h=5；

步骤二：循环预测

1.计算核权重

$$ K_h(18-x_i)=exp(-\frac{(18-x_i)^2}{2\times 5^2})$$

例如：
$$\begin{array}{l}
x_{1}=10: \exp \left(-\left(8^{2}\right) / 50\right) \approx 0.04 \\
x_{3}=20: \exp \left(-\left(2^{2}\right) / 50\right) \approx 0.92
\end{array}$$

2.权重归一化

权重总和：$ \sum K_h \approx 04+0.67+0.92+0.67+0.04=2.34。$

3.加权平均预测：

$$\hat{y}(18）=\frac{0.04×50+0.67×70+0.92×100+0.67×120+0.04×150}{2.34} \approx 95.2$$

##### 6.损失函数

上面的过程并没有提及损失函数，但事实上核回归的损失函数是均方误差，在确定核函数的前提下，优化核函数约等于优化带宽h。

~~~
# 手写函数实现
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义核函数：高斯核、Epanechnikov核、均匀核、三角核
def gaussian_kernel(u, h): 
    return np.exp(-(u ** 2) / (2 * h ** 2))
def epanechnikov_kernel(u, h): 
    return np.maximum(0, 1 - (u ** 2) / (h ** 2))
def uniform_kernel(u, h): 
    return np.where(np.abs(u) <= h, 1, 0)
def triangular_kernel(u, h): 
    return np.maximum(0, 1 - np.abs(u) / h)

# 2. 核回归预测
def kernel_regression(X_train, y_train, X_test, h=5.0, kernel='gaussian'):
    kernels = {
        'gaussian': gaussian_kernel,
        'epanechnikov': epanechnikov_kernel,
        'uniform': uniform_kernel,
        'triangular': triangular_kernel
    }
    kernel_func = kernels.get(kernel, gaussian_kernel)
    y_pred = np.zeros_like(X_test)
    for i, x in enumerate(X_test):
        weights = kernel_func(X_train - x, h)
        y_pred[i] = np.sum(weights * y_train) / np.sum(weights) if np.sum(weights) != 0 else np.mean(y_train)
    return y_pred

# 3. 测试数据
np.random.seed(42)
X_train = np.linspace(0, 10, 100)
y_train = np.sin(X_train) + np.random.normal(0, 0.1, 100)
X_test = np.linspace(0, 10, 200)

y_pred = kernel_regression(X_train, y_train, X_test, 5, 'gaussian')

# 5. 可视化
plt.scatter(X_train, y_train, s=10, alpha=0.6, label='Data')
plt.plot(X_test, y_pred, 'r-', label=f'Kernel Regression (h=5)')
plt.legend()
plt.show()
~~~

~~~
# sklearn实现，核回归+岭回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

# 生成数据
np.random.seed(42)
X_train = np.linspace(0, 10, 100).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, 100)

# 定义模型（使用RBF核，即高斯核）
"""
    rbf：即高斯核
    linear：线性核
    poly：多项式核
    sigmoid：sigmoid核
"""
model = KernelRidge(kernel='rbf')

# 网格搜索优化参数（alpha: 正则化强度, gamma: 核带宽的倒数）
params = {'alpha': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train, y_train)

# 最佳模型
best_model = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

# 预测
X_test = np.linspace(0, 10, 200).reshape(-1, 1)
y_pred = best_model.predict(X_test)

# 可视化
plt.scatter(X_train, y_train, s=10, alpha=0.6, label='Data')
plt.plot(X_test, y_pred, 'r-', label='Kernel Ridge Regression')
plt.legend()
plt.show()
~~~

