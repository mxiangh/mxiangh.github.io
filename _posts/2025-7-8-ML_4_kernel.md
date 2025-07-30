---
title: 机器学习（4）核回归——kernel Regression
tags: ML Regression
---

#### 核回归——Kernel Regression

思想：核回归是一种非参数回归方法，用于估计变量间的非线性关系，通过局部加权平均来拟合数据，权重由核函数决定。与线性回归不同，核回归不需要预设全局模型形式，而是通过数据驱动的方式灵活拟合局部特征。

<!--more-->

- 常见核函数：
  - 高斯核：$ K_h(u)=e^{-\frac{u^{2}}{2 h^{2}}} （常用）$
  - Epanechnikov核：$ K_h(u)=max(0,1-\frac{u^{2}}{h^{2}})$
  - 均匀核：$
  K_h(u)=\left\{\begin{array}{l}
  1, \text { if } \vert u \vert \le h \\
  0, \text { otherwise }
  \end{array}\right.
  $
  - 三角核：$ K_h(u)=max(0,1-\frac{\vert u \vert}{h})$

预测值计算公式：

$$\widehat{m}\left(x_{0}\right)=\widehat{E}\left[Y \mid X=x_{0}\right]=\frac{\sum_{i=1}^{n} K_{h}\left(x-x_{i}\right) y_{i}}{\sum_{i=1}^{n} K_{h}\left(x-x_{i}\right)}$$

- x：待预测的点；
- $x_i，y_i$：已知观测点的数据；
- K（·）：核函数。

带宽h：决定平滑程度。h越大，回归曲线越平滑（可能欠拟合）；h越小，对数据噪声越敏感（可能过拟合）。

核函数的主要作用：分配权重，核函数赋予距离x近的点更高的权重，远的点权重低。

算法流程：

（1）选择核函数与带宽

（2）循环预测点$x$

- 计算$x$与样本点$x_i$的距离；
  
- 将距离代入核函数中计算每个样本点的权重；
  
- 计算加权权重和；
  
- 计算预测值m($x$)

举个例子：

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