---
title: 机器学习（4）核回归——kernel Regression
tags: ML Regression
---

#### 核回归——Kernel Regression

思想：核回归是一种非参数回归方法，不需要事先假设回归函数的具体形式，通过局部加权平均来拟合复杂的非线性关系。

<!--more-->

常见核函数：
- 高斯核：$ K(u)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2} u^{2}} （常用）$
- Epanechnikov核：$ K(u)=\frac{3}{4}(1-u^{2})  for  \vert u \vert \leq 1 $
- 均匀核：$ K(u)=\frac{1}{2}  for  \vert u \vert \leq 1 $
- 三角核：$ K(u)=(1- \vert u \vert )  for  \vert u \vert \leq 1 $

预测值计算公式：

$$\widehat{m}\left(x_{0}\right)=\widehat{E}\left[Y \mid X=x_{0}\right]=\frac{\sum_{i=1}^{n} K_{h}\left(x_{i}-x_{0}\right) y_{i}}{\sum_{i=1}^{n} K_{h}\left(x_{i}-x_{0}\right)}$$


缩放：$ K_{h}(u)=\frac{1}{h} K\left(\frac{u}{h}\right) $

算法流程：

（1）选择核函数与带宽

（2）循环预测点$x_0$

a.计算$x_0$与样本点$x_i$的距离；
    
b.将距离代入核函数中计算每个样本点的权重；
    
c.计算加权权重和；
    
d.计算预测值m($x_0$)