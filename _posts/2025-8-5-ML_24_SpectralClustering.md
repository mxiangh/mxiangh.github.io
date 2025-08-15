---
title: 机器学习（24）谱聚类——Spectral Clustering
tags: ML Clustering
---

思想：基于图论和矩阵特征向量的聚类方法，将谱方法运用到聚类当中，先降维实现特征映射，在对映射后的数据运用传统聚类。

<!--more-->

谱聚类教学论文（2007）：A tutorial on spectral clustering

##### 1.图拉普拉斯矩阵的运用

在降维方法拉普拉斯特征映射LE中，算法通过构建拉普拉斯矩阵保持局部数据结构，将其映射到低维空间。

谱聚类为什么不直接用k-means聚类，而是在聚类前面加一步拉普拉斯特征映射，有以下几个解释：

（1）拉普拉斯矩阵能够捕捉数据的局部关系和全局流形结构，对非凸、螺旋交织数据等复杂的分布也能如此，相比于传统聚类对数据要求凸簇假设，运用范围更广；

（2）聚类本质是离散的优化问题，直接求解非常困难，但是拉普拉斯特征映射可以将离散向量转为连续的特征向量，降低计算复杂度，使得大规模数据聚类可行；

（3）加入拉普拉斯矩阵后，可解释性增强。

注：图拉普拉斯矩阵没有统一规定，作者说它是拉普拉斯矩阵，它就是拉普拉斯矩阵。

##### 2.算法步骤（拉普拉斯特征映射+K-Means聚类）

类似LE算法：

（1）构造权重矩阵W；

（2）根据权重矩阵W，构造度矩阵D和拉普拉斯矩阵L；

（3）找到拉普拉斯矩阵的前k个最大特征向量，按列构成X（k为簇的个数）；

聚类可使用K-Means聚类：

（4）对X进行聚类。

注：由于拉普拉斯矩阵的选取不同（归一化、非归一化），相似矩阵的构造方式不同（$\epsilon$领域、k近邻、全连接），算法细节可能略有不同，但总体结构如上。

~~~
~~~



~~~
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

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

# 数据标准化（谱聚类对特征尺度较敏感）
X_scaled = StandardScaler().fit_transform(X)

# 初始化并拟合谱聚类模型
spectral = SpectralClustering(
    n_clusters=2,
    affinity='rbf',  # 构建相似性矩阵，'rbf'：核函数，'nearest_neighbors'：k 近邻图
    gamma=0.1,      # 核函数参数，控制相似性的衰减速度
    random_state=42,
    n_init=10        # 多次初始化k-means选择初始质心，选择这 10 次结果中目标函数（如簇内平方和）最小的聚类结果作为最终输出
)
labels = spectral.fit_predict(X_scaled)

# 获取真实标签用于对比（第4列）
true_labels = data[:, 3]

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

plt.title('Spectral Clustering on Watermelon Dataset')
plt.xlabel('Density')
plt.ylabel('Sugar Content')
plt.legend()
plt.show()

~~~

