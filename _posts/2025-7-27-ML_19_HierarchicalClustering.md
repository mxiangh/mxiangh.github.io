---
title: 机器学习（19）层次聚类——Hierarchical Clustering
tags: ML Clustering
---

思想：一种无监督的聚类算法，使用距离衡量两个簇的相似度，通过递归地对数据进行合并（或拆分），构建一个类似树的聚类结构，称为“树状图”（Dendrogram）。

<!--more-->

##### 1.算法类别

层次聚类有两种类型，一种是“自底向上”的算法：凝聚层次聚类（Agglomerative Hierarchical Clustering）；另一种是“自顶向下”的算法：分裂层次聚类（Divisive Hierarchical Clustering）。

（1）凝聚层次聚类

这是最常用的算法：

- “自底向上”的方法，初始阶段把每个样本都看成一个簇；
- 之后每一步根据簇的距离，来合并最相似的两个簇，直到全都变成一个簇，或者达到预设的停止条件；
- 一般要构建树状图。

（2）分裂层次聚类

不常用的算法，因为计算复杂度高：

- “自顶向下”的方法，最初将所有数据视为一个簇；
- 然后递归地将簇分裂成两个较小的簇，直到每个簇只包含一个数据点或达到预设的停止条件；

##### 2.层次聚类原理

层次聚类的核心是如何定义簇之间的距离来衡量簇之间的相似性。

（1）距离度量

一般常用欧式距离，此外还可选择曼哈顿、闵可夫斯基。

（2）簇间距离度量方法

单链接（Single Linkage）：由两个簇最近的样本点距离决定。

$$ d_{min}(C_i,C_j)=\underset{\mathbf{x} \in C_i,\mathbf{z} \in C_j}{min} dist(\mathbf{x},\mathbf{z}) $$

全链接（Complete Linkage）：由两个簇最远的样本点距离决定。

$$ d_{max}(C_i,C_j)=\underset{\mathbf{x} \in C_i,\mathbf{z} \in C_j}{max} dist(\mathbf{x},\mathbf{z}) $$

均链接（Average Linkage）：由两个簇所有样本点的平均距离决定。

$$ d_{avg}(C_i,C_j)=\frac{1}{\vert C_i \vert \vert C_j \vert} \sum_{\mathbf{x} \in C_i} \sum{\mathbf{z} \in C_j} dist(\mathbf{x},\mathbf{z}) $$

质心链接（Centroid Linkage）：由两个簇的质心距离决定。

$$ d_{cen}(C_i,C_j)=\vert \vert \mu_{C_i} - \mu_{C_j} \vert \vert $$

##### 3.算法步骤（凝聚层次聚类）

（1）计算数据集中每两个点之间的距离，将距离存入矩阵中。常用的距离度量是欧氏距离。

（2）初始化每个点为一个单独的簇。

（3）查找距离矩阵中距离最近的两个簇，合并这两个簇，并更新簇之间的距离。

（4）更新距离矩阵：更新新的簇与其他簇之间的距离。

注：不同的簇间距离计算方式会影响聚类结果，常用的计算方式包括单链（最小距离）、全链（最大距离）、平均距离和质心距离。

（5）重复步骤 3 和 4，直到所有点合并成一个簇或达到预设的聚类数。

（6）树状图的生成
在合并簇的过程中，记录每次合并的簇对及其距离，即可构建一个树状图（dendrogram），展示数据集在不同距离阈值下的聚类结构。

##### 4.层次聚类的优缺点

优点

- 无需预先指定簇数：层次聚类在生成树状图后，可以按需选择不同的层次切割位置，从而得到不同数量的簇。
- 适用于不同规模的簇：层次聚类可以适用于不规则形状的簇，尤其是在选择单链或平均距离时。
- 层次性：层次聚类提供了数据的层次结构，使得不同粒度的聚类信息都能展现，适合可视化。
- 可解释性：通过树状图直观展示聚类过程和结构，便于解释数据点之间的层次关系。

缺点

- 计算复杂度高：层次聚类的时间复杂度通常为 ，在大数据集上计算代价很高，通常只适合较小的数据集。
- 难以调整合并错误：一旦在某一层次上错误地合并了两个簇，这个错误会一直影响后续层次，无法在后续步骤中调整。
- 对噪声和离群点敏感：层次聚类在没有噪声过滤的情况下，可能会将离群点视为一个单独的簇或加入其他簇，从而影响聚类质量。
- 不同链接方法的局限性：不同的链接方法（单链、全链等）对聚类结果影响较大，且可能导致链式聚类等问题，选择合适的方法需要根据数据特性调整。

~~~

~~~



~~~
# scipy库实现（绘制树状图）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = ["SimHei"]  # 支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 西瓜数据集4.0
data = {
    '编号': list(range(1, 31)),
    '密度': [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 
             0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719, 0.359, 
             0.339, 0.282, 0.748, 0.714, 0.483, 0.478, 0.525, 0.751, 0.532, 
             0.473, 0.725, 0.446],
    '含糖率': [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 
              0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103, 0.188, 
              0.241, 0.257, 0.232, 0.346, 0.312, 0.437, 0.369, 0.489, 0.472, 
              0.376, 0.445, 0.459],
    '好瓜': ['是', '是', '是', '是', '是', '是', '是', '是', '否', 
           '否', '否', '否', '否', '否', '否', '否', '否', '否', 
           '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '是', '是']
}

# 创建DataFrame
df = pd.DataFrame(data)

# 提取特征并标准化
X = df[['密度', '含糖率']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

methods=['single', 'complete', 'average', 'ward']
method_names=['单链距离', '全链距离', '平均链距离', '沃德距离']
Z = linkage(X_scaled, method=methods[3])

dendrogram(Z, labels=df['编号'].values)
plt.title(f'层次聚类树状图 - {method_names[3]}')
plt.xlabel('西瓜编号')
plt.ylabel('距离')
plt.show()

# 根据树状图，选择合适的阈值进行聚类，这里选择3个簇
clusters = fcluster(Z, t=3, criterion='maxclust')

# 绘制聚类散点图
scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=100, alpha=0.7)
plt.title(f'西瓜数据集4.0层次聚类结果 - {method_names[3]}')
plt.xlabel('密度')
plt.ylabel('含糖率')

# 添加数据点编号
for j, txt in enumerate(df['编号']):
    plt.annotate(txt, (X[j, 0], X[j, 1]), fontsize=12)

# 添加颜色条
plt.colorbar(scatter, label='簇编号')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 比较聚类结果与实际标签
print("\n聚类结果与实际标签比较：")
comparison_df = df[['编号', '好瓜']].copy()

Z_best = linkage(X_scaled, method = methods[3])
# 和前面聚类一样这里只是将簇改成2
comparison_df['聚类结果'] = fcluster(Z_best, t=2, criterion='maxclust')  
print(comparison_df)

~~~

