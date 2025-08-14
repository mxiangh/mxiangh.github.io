---
title: 机器学习（23）均值漂移——MeanShift
tags: ML Clustering
typora-root-url: ./..
---

思想：基于密度的聚类方法，不需要事先设定簇的个数，根据每个数据点的偏移方向找到密度最高的区域，将区域相同的数据点合并为一个簇。

<!--more-->

论文：The Estimation of the Gradient of a Density Function, with Applications in Pattern Recognition

算法改进：Mean Shift, Mode Seeking, and Clustering

##### 1.基本原理

Mean Shift目的是找到从所有数据点开始形成的密度最大的区域，将区域重合的点合并成为一个簇。

在初始化时，随机选择一个数据点，以该数据点为圆心，设定一个半径，例如下图中的C1_o就是初始选的随机点，C1是初始画的圆。

![](/assets/images/MeanShift/one.png)

之后，计算初始圆内所有样本的均值（图中的C1_r），圆心到均值的方向表明的是当前区域密度最大的方向，所以让圆心朝着该方向移动，形成第二个圆C2。

重复进行，计算质心，移动圆，直到质心不再变化或者达到阈值则停止。这样，一个数据点的密度最高区域就找到了。

为了找到每个数据点的密度最高区域，需要对每个数据点实现上述步骤：画圆、朝着高密度区域移动圆、质心不变或收敛。

最后，将最高密度区域相同的数据点合并为一个簇，所以该方法不需要实现确定簇的个数。

##### 2.数学公式

2.1 均值漂移

$$M(x)=\frac{\sum_{s \in S} K(s - x)s}{\sum_{s \in S} K(s - x)} \tag{2}$$

$$K(x)=\begin{cases} 1, & \text{若} \Vert x \Vert \leq \lambda \\ 0, & \text{若} \Vert x \Vert > \lambda \end{cases} \tag{1}$$
x为初始选择的样本，s为其他样本，$\lambda$是是一个阈值，在这里可以理解为半径。$K(s - x)$用来找以x为圆心，$\lambda$为半径的样本，由于值取0和1，分子可以看做在圆内的样本向量求和，分母为圆内样本总数量，对公式化简如下

$$ M_t(x)=\frac{1}{k}\sum_{x_i \in S_h} (x_i-x_t) $$

$$S_h(x) = \left\{ y \mid (y - x)(y - x)^T \leqslant h^2 \right\}$$

- $S_h$：以$x_t$为中心点，半径为h的高维球区域，例如上面的蓝色圆。也可以理解为所有满足y 到 x 的欧氏距离的平方不超过$h^2$的点 y 的集合。
- $k$：包含在$S_h$范围内的数据点个数。
- $x_i$：包含在$S_h$范围内的数据点向量。

计算邻域内所有点到当前中心点的偏差$(x_i-x) $，对所有偏差求均值，得到中心点的漂移方向$M_t$。如果$M_t$远离当前中心点$x$，说明该区域的密度中心在别处，需要调整位置。

2.2 调整中心位置

$$x_{t+1}=M_t-x_t$$

当前中心点$x_t$沿着均值偏移量$M_t$方向移动，得到新的中心$x_{t+1}$，该迭代过程不断调整中心点位置，直到质心不变或收敛。

##### 3.Mean Shift改进

原始Mean Shift有两个问题：

（1）原始Mean Shift常基于平坦核，这类核在处理数据时，对邻域内样本 “一视同仁”，不管样本离被偏移点远近，只要在核定义的邻域内，贡献就相同，无法精准体现不同距离样本对中心更新的不同影响。

（2）原始Mean Shift未充分考虑样本权重，不能根据样本与被偏移点的关联程度（距离 ）差异化调整贡献。

于是有了新的均值漂移公式：

$$M(x) = \frac{\sum_{s \in S} K(s - x)w(s)s}{\sum_{s \in S} K(s - x)w(s)}$$

其中$K$是高斯核，$S$是数据集，$x$是被偏移点，$w(s)$是数据点s的权重，之后通过$m(x) - x$进行均值偏移。

~~~
~~~



~~~
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# 西瓜数据集4.0
data = np.array([
    [1, 0.697, 0.460, 1],[2, 0.774, 0.376, 1],[3, 0.634, 0.264, 1],[4, 0.608, 0.318, 1],[5, 0.556, 0.215, 1],
    [6, 0.403, 0.237, 0],[7, 0.481, 0.149, 0],[8, 0.437, 0.211, 0],[9, 0.666, 0.091, 0],[10, 0.243, 0.267, 0],
    [11, 0.245, 0.057, 0],[12, 0.343, 0.099, 0],[13, 0.639, 0.161, 1],[14, 0.657, 0.198, 1],[15, 0.360, 0.370, 0],
    [16, 0.593, 0.042, 0],[17, 0.719, 0.103, 1],[18, 0.359, 0.188, 0],[19, 0.339, 0.241, 0],[20, 0.282, 0.257, 0],
    [21, 0.748, 0.232, 1],[22, 0.714, 0.346, 1],[23, 0.483, 0.312, 0],[24, 0.478, 0.437, 0],[25, 0.525, 0.369, 1],
    [26, 0.751, 0.489, 1],[27, 0.532, 0.472, 1],[28, 0.473, 0.376, 0],[29, 0.725, 0.445, 1],[30, 0.446, 0.459, 0]
])

# 提取密度和含糖率作为特征（第2和第3列）
X = data[:, 1:3]

# 估计带宽参数，这是Mean Shift的关键参数
# quantile参数控制用于估计带宽的点的比例，值越小带宽越小，聚类数量可能越多
bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=len(X))

# 初始化并拟合Mean Shift模型
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

# 获取聚类标签和中心点
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# 计算聚类数量
n_clusters_ = len(np.unique(labels))
print(f'估计的聚类数量: {n_clusters_}')

# 可视化结果
plt.figure(figsize=(8, 6))

# 使用不同颜色绘制不同聚类
colors = plt.cm.get_cmap('tab10', n_clusters_).colors
for i in range(n_clusters_):
    cluster_data = X[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=100, 
                c=[colors[i]], label=f'Cluster {i + 1}')

# 绘制聚类中心点
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
            c='red', marker='x', s=200, linewidths=3, label='Centers')

plt.title('Mean Shift Clustering')
plt.xlabel('Density')
plt.ylabel('Sugar Content')
plt.legend()
plt.show()

~~~

