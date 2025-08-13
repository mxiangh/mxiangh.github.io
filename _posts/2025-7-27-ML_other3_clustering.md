---
title: 机器学习（3）聚类简介
tags: ML Tips Clustering
---

简单介绍一下聚类

<!--more-->

聚类通过将数据划分成不同的簇对样本进行分类。簇指的是类别，将数据划分成多个簇指的是将数据分为多个类别，想分成几类，就需要找几个簇。

聚类评估指标分为两类：“外部指标”和“内部指标”。

1.外部指标：将聚类结果与某个“参考模型”进行比较。

$$对数据集D= \lbrace \mathbf{x_1} , \mathbf{x_2} , \cdots , \mathbf{x_m} \rbrace，假定通过聚类给出的簇划分为C = \lbrace C_1, C_2, \cdots ,C_k  \rbrace ，参考模型给出的簇划分为C^{*} = \lbrace C_{1}^{*} , C_{2}^{*} , \cdots , C_{k}^{*} \rbrace。相应地，令\lambda 和\lambda^{*} 分别表示与C和C^{*} 对应的簇标记向量，将样本两辆配对考虑，定义$$

$$\begin{array}{ll}
a=|S S|, & \left.S S=\left\{\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right) \mid \lambda_{i}=\lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\right)\right\} \\
b=|S D|, & \left.S D=\left\{\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right) \mid \lambda_{i}=\lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\right)\right\} \\
c=|D S|, & \left.D S=\left\{\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right) \mid \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\right)\right\} \\
d=|D D|, & \left.D D=\left\{\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right) \mid \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\right)\right\}
\end{array}$$

其中：

集合$SS$包含了在$C$中隶属于相同簇且在$C^{*}$中也隶属于相同簇的样本对；

集合$SD$包含了在$C$中隶属于相同簇但在$C^{*}$中隶属于不相同簇的样本对；

集合$DS$包含了在$C$中隶属于不相同簇但在$C^{*}$中隶属于相同簇的样本对；

集合$DD$包含了在$C$中隶属于不相同簇且在$C^{*}$中也隶属于不相同簇的样本对。

由于每个样本对$( \mathbf{x}_i, \mathbf{x}_j ) (i<j) $仅能出现在一个集合中，因此$ a+b+c+d = \frac{m(m-1)}{2} $。

- Jaccard系数（Jaccard Coefficient，简称JC）

$$ JC=\frac{a}{a+b+c} $$

- FM指数（Fowlkes and Mallows Index，简称FMI）

$$ FMI=\sqrt{\frac{a}{a+b}\frac{a}{a+c}} $$

- Rand指数（Rand Index，简称RI）

$$ RI=\frac{2(a+d)}{m(m-1)}$$

上述指标值越大越好。

2.内部指标：直接考察聚类结果而不用任何参考模型。

对于聚类结果簇划分$C=\lbrace C_1, C_2,\cdots,C_k \rbrace$，定义

$$ avg(C)=\frac{2}{\vert C \vert (\vert C \vert - 1)} \sum_{1 \leq i <j \leq \vert C \vert } dist(\mathbf{x}_{i}, \mathbf{x}_{j})$$

$$ diam(C)=\underset{1 \leq i <j \leq \vert C \vert }{max} dist(\mathbf{x}_{i}, \mathbf{x}_{j})$$

$$ d_{min}(C_i,C_j)=\underset{\mathbf{x}_i \in C_i,\mathbf{x}_j \in C_j}{min} dist(\mathbf{x}_{i}, \mathbf{x}_{j})$$

$$ d_{cen}(C_i,C_j)=dist(\mu_i,\mu_j)$$

其中，dist用于计算两个样本之间的距离，$\mu$代表簇$C$的中心点$\mu=\frac{1}{\vert C \vert}\sum_{1 \leq i \leq \vert C \vert} \mathbf{x}_i$。

$avg(C)$对应于簇C内样本间的平均距离；

$diam(C)$对应于簇内样本间的最远距离；

$d_{min}(C_i,C_j)对应于$簇$C_i$与簇$C_j$最近样本间的距离；

$d_{cen}(C_i,C_j)$对应于$簇$C_i$与簇$C_j$中心点间的距离。

- DB指数（Davies-Bouldin Index，简称DBI）

$$ DBI=\frac{1}{k} \sum_{i=1}^k \underset{j \ne i}{max} (\frac{avg(C_i)+avg(C_j)}{d_{cen}(\mu_i,\mu_j)})$$


- Dunn指数（Dunn Index，简称DI）

$$ DI=\underset{1 \leq i \leq k}{min} \lbrace \underset{j \ne i}{min} (\frac{d_{min}(C_i,C_j)}{\underset{1 \leq l \leq k}{max}\ diam(C_l)}) \rbrace $$

DBI越小越好，DI越大越好。

有序属性选择闵可夫斯基距离，无需属性用VDM距离，两者结合可以处理混合属性。

