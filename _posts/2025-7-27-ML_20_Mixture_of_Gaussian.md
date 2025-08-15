---
title: 机器学习（20）高斯混合聚类——Gaussian Mixture Model, GMM
tags: ML Clustering
---

思想：使用高斯分布和贝叶斯定理，为每个样本确定后验概率，根据后验概率划分簇。

<!--more-->

##### 1.高斯混合分布定义

先回顾一下一元高斯分布概率密度函数

$$ f(x)=\frac{1}{\sqrt{2 \pi} \sigma} exp(-\frac{(x-\mu)^2}{2 \sigma^2}) $$

现在给出一般的多元高斯分布概率密度函数

$$ p(x)=\frac{1}{(2 \pi)^{\frac{n}{2}} \vert \Sigma \vert^{\frac{1}{2}}} exp(-\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) ) $$

其中，$\mu$是n维均值向量，$\Sigma$是$n \times n$的协方差矩阵。一元高斯分布是多元高斯分布当n取1时的特例，高斯分布完全由均值向量$\mu$和协方差矩阵$\Sigma$这两个参数确定。将概率密度函数记为$p(x\mid \mu, \Sigma)$，可以理解为条件概率，当$\mu$和$\Sigma$取某值时，$p(x)$的概率。

现在可以定义高斯混合分布

$$ p_M(x)=\sum_{i=1}^k \alpha_i \cdot p(x\mid \mu_i, \Sigma_i) $$

该分布由k个混合成分组成，每个混合成分对应一个高斯分布。其中$ \mu_i$ 与$ \Sigma_i$是第i个高斯混合成分的参数，$\alpha_i >0$为相应的“混合系数”，$\sum_{i=1}^k \alpha_i=1$。

理解“混合成分”：假设我们有一个包含人的身高体重数据的数据集，它可能不是简单的单一高斯分布形态。但可以想象这个数据是由几个不同群体（比如不同性别群体、不同年龄段群体等）的数据混合而成， 每个群体内的数据分布可以近似看作是一个高斯分布，那么这些不同群体对应的高斯分布就可以理解为高斯混合分布中的混合成分。通过调整各个混合成分的参数（均值、协方差）和混合系数，就能让高斯混合分布更好地拟合这个复杂的身高体重数据分布。

##### 2.贝叶斯定理的应用

回顾朴素贝叶斯定理涉及到的概念：

（1）先验概率：基于以往经验、历史数据或主观判断对事件发生概率的初始估计。

例如：随机抽一张扑克牌是红桃的概率为 13/52=25%。

（2）后验概率（条件概率）：事情已经发生，要求这件事情发生的原因是由某个因素引起的可能性的大小。

例如：已知抽到的牌是红色，则 “这张牌是红桃” 的概率为13/26=50%（红色牌中红桃占一半）。

$P(A \vert B)$是已知B发生后A的条件概率，也由于得自B的取值而被称作A的后验概率。

$ P(B \vert A) $ 是已知A发生后B的条件概率，也由于得自A的取值而被称作B的后验概率。

（3）联合概率：表示两件事共同发生（数学概念上的交集）的概率，A和B的联合概率表示为$P(A \cap B)$或$P(A , B)$或$P(AB)$。

（4）全概率公式：$P(A)=\sum_{i = 1}^{n}P(A\mid B_i)P(B_i)$，把事件 A 发生的概率，通过样本空间的划分，分解为在不同条件 $B_i$ 下 A 发生的概率的加权和，权重就是条件 $B_i$ 本身发生的概率。

（5）贝叶斯定理

$$ P(Y \vert X)=\frac{P(X \vert Y) P(Y)}{P(X)} $$

##### 3.高斯混合分布+贝叶斯定理构建损失函数

在朴素贝叶斯里面说过，贝叶斯定理一般用于已知三个概率，求第四个概率。而高斯混合分布恰巧也满足这一点，把$\alpha$看成样本是某个高斯混合的先验概率，现在要求的是在已知样本的前提下样本属于某个高斯混合成分的条件概率（后验概率）。

假设训练集$D=\lbrace \mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_m \rbrace$，令随机变量$z_j \in \lbrace 1,2\cdots,k \rbrace$表示生成样本$\mathbf{x}_j$的高斯混合成分，$z_j$的先验概率$P(z_j=i)$对应于$\alpha_i(i=1,2,··.,k)$。根据贝叶斯定理，$z_j$的后验分布对应于

$$ \begin{aligned}
p_M(z_j=i \mid \mathbf{x}_j) & = \frac{P(z_j=i)\cdot p_M(\mathbf{x}_j \mid z_j=i)}{p_M(\mathbf{x}_j)} \\
& = \frac{\alpha_i \cdot p(x_j\mid \mu_i, \Sigma_i)}{\sum_{i=1}^k \alpha_i \cdot p(x_j\mid \mu_i, \Sigma_i)}
\end{aligned}$$

$p_M(z_j=i \mid \mathbf{x}_j)$是样本$\mathbf{x}_j$由第i个高斯混合成分生成的后验概率。

接着把样本集D划分成K个簇$C=\lbrace C_1,C_2,\cdots,C_m \rbrace$，每个样本$\mathbf{x}_j$的簇标记$\lambda_j$为

$$ \lambda_j=\underset{i \in \lbrace1,2,\cdots,k\rbrace}{arg\ max}\ p_M(z_j=i \mid \mathbf{x}_j) $$

##### 4.最大期望算法（Expectation-maximization algorithm，EM算法）

为了求解上述模型，首先考虑极大似然估计，因为极大似然估计通常用来求概率最大

$$ \begin{aligned}
LL(D) & =ln(\prod_{j=1}^{m}p_M(\mathbf{x_j}) \\
& = \sum_{j=1}^{m} ln(\sum_{i=1}^k \alpha_i \cdot p(x_j\mid \mu_i, \Sigma_i)
\end{aligned}$$

4.1 推导：

（1）$\mu_i$求解

上述极大似然估计函数有三个未知数，求$(\alpha_i,\mu_i,\Sigma_i)$使得似然估计最大，对$\mu_i$求偏导并为0

$$\begin{aligned}
\frac{\partial LL(D)}{\partial \mu_i} & = \sum_{j=1}^m \frac{\partial}{\partial \mu_i} \ln (\sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mu_l, \Sigma_l ) \\
& = \sum_{j=1}^m \frac{1}{\sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mu_l, \Sigma_l)} \cdot \frac{\partial}{\partial \mu_i} ( \sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mu_l, \Sigma_l ) \\
& = \sum_{j=1}^m \frac{1}{\sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mu_l, \Sigma_l)} \cdot \alpha_i \cdot \frac{\partial p(\mathbf{x}_j \mid \mu_i, \Sigma_i)}{\partial \mu_i}
\end{aligned}$$

其中

$$ p(\mathbf{x}_j \mid \mu_i, \Sigma_i) = \frac{1}{(2 \pi)^{\frac{d}{2}} \vert \Sigma_i \vert^{\frac{1}{2}}} exp(-\frac{1}{2} (\mathbf{x}_j-\mu_i)^T \Sigma_i^{-1} (\mathbf{x}_j-\mu_i) ) $$

则有

$$ \frac{\partial p(\mathbf{x}_j \mid \mathbf{\mu}_i, \mathbf{\Sigma}_i)}{\partial \mathbf{\mu}_i} = p(\mathbf{x}_j \mid \mathbf{\mu}_i, \mathbf{\Sigma}_i) \cdot \mathbf{\Sigma}_i^{-1} (\mathbf{x}_j - \mathbf{\mu}_i) $$

代入

$$ \frac{\partial LL(D)}{\partial \mathbf{\mu}_i} = \sum_{j=1}^m \frac{1}{\sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mathbf{\mu}_l, \mathbf{\Sigma}_l)} \cdot \alpha_i \cdot p(\mathbf{x}_j \mid \mathbf{\mu}_i, \mathbf{\Sigma}_i) \cdot \mathbf{\Sigma}_i^{-1} (\mathbf{x}_j - \mathbf{\mu}_i)=0 $$

注意，这里的$\mathbf{\Sigma}_i$是可逆矩阵，若可逆矩阵乘以向量等于 0，则向量本身必须等于 0。所以上述式子等价于

$$ \sum_{j=1}^m \frac{\alpha_i \cdot p(\mathbf{x}_j \mid \mathbf{\mu}_i, \mathbf{\Sigma}_i)}{\sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mathbf{\mu}_l, \mathbf{\Sigma}_l)} \cdot (\mathbf{x}_j - \mathbf{\mu}_i)=0 $$

由3提出的后验概率并令$\gamma_{ji}=p_M(z_j=i \mid \mathbf{x}_j)$，有

$$ \mathbf{\mu}_i = \frac{\sum_{j=1}^m \gamma_{ji} \mathbf{x}_j}{\sum_{j=1}^m \gamma_{ji}} $$

也就是说，混合成分的均值可以通过样本加权平均来估计，样本权重是每个样本属于该成分的后验概率。

（2）$\Sigma_i$求解

类似地，对$\Sigma_i$求偏导取0

$$\begin{aligned}
\frac{\partial LL(D)}{\partial \mathbf{\Sigma}_i} & = \sum_{j=1}^m \frac{\partial}{\partial \mathbf{\Sigma}_i} \ln (\sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mu_l, \Sigma_l ) \\
& = \sum_{j=1}^m \frac{1}{\sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mu_l, \Sigma_l)} \cdot \frac{\partial}{\partial \mathbf{\Sigma}_i} ( \sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mu_l, \Sigma_l ) \\
& = \sum_{j=1}^m \frac{1}{\sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mu_l, \Sigma_l)} \cdot \alpha_i \cdot \frac{\partial p(\mathbf{x}_j \mid \mu_i, \Sigma_i)}{\partial \mathbf{\Sigma}_i}
\end{aligned}$$

将$p(\mathbf{x}_j \mid \mu_i, \Sigma_i)$记为 p，令 $\mathbf{\Sigma} = \Sigma_i$，$\mathbf{\mu} = \mu_i$，$\mathbf{x} = \mathbf{x}_j$，对p取对数并对$\Sigma$求偏导

$$ \ln p = -\frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln|\mathbf{\Sigma}| - \frac{1}{2}(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu}) $$

$$ \frac{\partial \ln p}{\partial \mathbf{\Sigma}} = -\frac{1}{2}\mathbf{\Sigma}^{-1} + \frac{1}{2}\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{\mu})(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} $$

又因为$ \frac{\partial \ln p}{\partial \mathbf{\Sigma}} = \frac{\partial \ln p}{\partial p}\frac{\partial p}{\partial \mathbf{\Sigma}} = \frac{1}{p} \frac{\partial p}{\partial \mathbf{\Sigma}} $，所以

$$ \frac{\partial p(\mathbf{x}_j\mid\mu_i, \mathbf{\Sigma}_i)}{\partial \mathbf{\Sigma}_i} = p(\mathbf{x}_j\mid\mu_i, \mathbf{\Sigma}_i) \lbrack -\frac{1}{2} \mathbf{\Sigma}_i^{-1} + \frac{1}{2} \mathbf{\Sigma}_i^{-1}(\mathbf{x}_j - \mu_i)(\mathbf{x}_j - \mu_i)^T \mathbf{\Sigma}_i^{-1} \rbrack $$

代入

$$ \frac{\partial LL(D)}{\partial  \mathbf{\Sigma}_i} = \sum_{j=1}^m \frac{1}{\sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mathbf{\mu}_l, \mathbf{\Sigma}_l)} \cdot \alpha_i \cdot p(\mathbf{x}_j\mid\mu_i, \mathbf{\Sigma}_i) \lbrack -\frac{1}{2} \mathbf{\Sigma}_i^{-1} + \frac{1}{2} \mathbf{\Sigma}_i^{-1}(\mathbf{x}_j - \mu_i)(\mathbf{x}_j - \mu_i)^T \mathbf{\Sigma}_i^{-1} \rbrack = 0 $$

与4.2一样，令$\gamma_{ji}=p_M(z_j=i \mid \mathbf{x}_j)$，有

$$ \Sigma_i = \frac{\sum_{j=1}^m \gamma_{ji} (\mathbf{x}_j - \mathbf{\mu}_i)(\mathbf{x}_j - \mathbf{\mu}_i)^T}{\sum_{j=1}^m \gamma_{ji}} $$

（3）$\alpha_i$求解

对于混合系数$\alpha_i$，除了最大化$LL(D)$，还需满足$\alpha_i \geq 0$，$\sum_{i=1}^k \alpha_i = 1$，所以考虑拉格朗日乘子法求解

$$ LL(D)+\lambda(\sum_{i=1}^k \alpha_i - 1) $$

令式子对$\alpha_i$并取0

$$ \sum_{j=1}^m \frac{ p(\mathbf{x}_j \mid \mathbf{\mu}_i, \mathbf{\Sigma}_i)}{\sum_{l=1}^k \alpha_l \cdot p(\mathbf{x}_j \mid \mathbf{\mu}_l, \mathbf{\Sigma}_l)} + \lambda = 0$$

两边同乘$\alpha_i$，可得

$$ \sum_{j=1}^m \gamma_{ji} + \lambda \alpha_i = 0 $$

由于$ \gamma_{ji}$表示第$j$个样本属于第$i$个高斯成分的概率，对所有样本求和有

$$\sum_{i=1}^k \gamma_{ji} = 1$$

同时有约束

$$\sum_{i=1}^k \alpha_i = 1$$

所以两边式子对所有样本求和

$$ \begin{aligned}
\sum_{j=1}^m \sum_{i=1}^k \gamma_{ji} + \lambda \sum_{i=1}^k \alpha_i & = 0 \\
\sum_{j=1}^m 1 + \lambda & = 0 \\
\lambda & = -m
\end{aligned}$$

代入求导后的拉格朗日函数可得

$$ \alpha_i = \frac{1}{m} \sum_{j=1}^m \gamma_{ji} $$

4.2 EM算法

之后通常使用EM算法求解。

解释：在概率模型中寻找参数最大似然估计或者最大后验估计的算法，其中概率模型依赖于无法观测的隐性变量。

步骤：

（1）第一步是计算期望（E），利用对隐藏变量的现有估计值，计算其最大似然估计值；

（2）第二步是最大化（M），在E步上求得的最大似然值来计算参数的值。M步上找到的参数估计值被用于下一个E步计算中，这个过程不断交替进行。

根据上述步骤，第一步，可以先初始化三个未知参数$(\alpha_i,\mu_i,\Sigma_i)$的值得到后验概率$ \gamma_{ji}$；

第二步，根据得到的后验概率，更新三个未知参数。

重复这两步，直到达到最大迭代轮数，或者似然函数增长很少甚至不增长。

##### 5.优缺点

优点

- 通过估计高斯分布的参数（均值、协方差矩阵等）来描述数据的分布特性，能够很好地拟合数据的概率分布。
- 能够捕捉到数据中的多峰结构，对于具有多个不同类别的复杂数据集，能够有效地将它们区分开来。
- 可以为每个数据点提供属于各个聚类的概率，这种概率表示方式提供了更多的信息。

缺点

- 高斯混合聚类的计算复杂度相对较高，尤其是在处理大规模数据集时。
- 高斯混合模型的参数估计通常依赖于初始值的选择，不同的初始值可能会导致算法收敛到不同的局部最优解。
- 需要事先确定高斯分布的数量（即聚类的类别数），但在实际应用中，数据集的真实类别数往往是未知的。

~~~

~~~



~~~
# sklearn实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 提供的西瓜数据集4.0
data = np.array([
    [1, 0.697, 0.460, 1],[2, 0.774, 0.376, 1],[3, 0.634, 0.264, 1],[4, 0.608, 0.318, 1],[5, 0.556, 0.215, 1],
    [6, 0.403, 0.237, 0],[7, 0.481, 0.149, 0],[8, 0.437, 0.211, 0],[9, 0.666, 0.091, 0],[10, 0.243, 0.267, 0],
    [11, 0.245, 0.057, 0],[12, 0.343, 0.099, 0],[13, 0.639, 0.161, 1],[14, 0.657, 0.198, 1],[15, 0.360, 0.370, 0],
    [16, 0.593, 0.042, 0],[17, 0.719, 0.103, 1],[18, 0.359, 0.188, 0],[19, 0.339, 0.241, 0],[20, 0.282, 0.257, 0],
    [21, 0.748, 0.232, 1],[22, 0.714, 0.346, 1],[23, 0.483, 0.312, 0],[24, 0.478, 0.437, 0],[25, 0.525, 0.369, 1],
    [26, 0.751, 0.489, 1],[27, 0.532, 0.472, 1],[28, 0.473, 0.376, 0],[29, 0.725, 0.445, 1],[30, 0.446, 0.459, 0]
])

# 提取密度和含糖率作为特征
X = data[:, 1:3]

# 聚类结果
k = 3  # 假设分为3个簇

gmm = GaussianMixture(n_components=k, random_state=42)
gmm.fit(data)

labels = gmm.predict(data)

# 可视化结果
plt.figure(figsize=(10, 6))

# 自动生成颜色
colors = plt.cm.get_cmap('tab10', k).colors

# 可视化结果
plt.figure(figsize=(8, 6))
for i in range(k):
    # 选择当前聚类的样本
    cluster_data = data[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=100, label=f'Cluster {i + 1}')

# 绘制高斯分布的中心点
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')

plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Density')
plt.ylabel('Sugar Content')
plt.legend()
plt.show()

~~~

