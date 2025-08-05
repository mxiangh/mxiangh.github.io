---
title: 机器学习补充（2）KL散度与交叉熵
tags: ML Tips
---

介绍一下，熵、KL散度、交叉熵。

<!--mare-->

#### 1.熵（Entropy）：衡星分布的不确定性

在信息论中，嫡用于描述一个概率分布的不确定性，熵越大，分布的不确定性越高。对于离散型随机变量  Y  ，其概率分布为  $P(Y=y)  $，熵的定义为：

$$H(P)=-\sum_{y} P(y) \log P(y)$$


例如，二分类中 $ Y \in\{0,1\} $ ，真实分布 $ P(Y=1)=p, P(Y=0)=1-p  $，则镝为：

$$H(P)=-[p \log p+(1-p) \log (1-p)]$$

#### 2.相对嫡（KL 散度）：衡星两个分布的差异

当我们需要比较两个概率分布  P （真实分布）和  Q （模型预测分布）的差异时，常用KL散度（Kullback－Leibler Divergence）来度量。其定义为：

$$K L(P \vert \vert Q)=\sum_{y} P(y) \log \frac{P(y)}{Q(y)}$$


KL 散度的性质：
- 非负性：$ K L(P \vert \vert  Q) \geq 0  $，当且仅当 $ P=Q  $时取等号（此时差异为 0 ）；
- 不对称性：$ K L(P\vert \vert  Q) \neq K L(Q \| P) $ ，但核心是衡量＂用  Q  近似  P  时的信息损失＂。

#### 3.交叉嫡（Cross－Entropy）：KL 散度的简化

将 KL 散度展开：

$$\begin{aligned}
KL(P \vert \vert  Q) & =\sum_{y} P(y) \log P(y)-\sum_{y} P(y) \log Q(y) \\
& = -H(P)+\left(-\sum_{y} P(y) \log Q(y)\right)
\end{aligned}$$


其中， H(P)  是真实分布  P  的熵（与模型无关，为常数），因此最小化 KL 散度等价于最小化交叉嫡：

$$H(P, Q)=-\sum_{y} P(y) \log Q(y)$$


这就是交叉嫡的定义。在机器学习中，我们希望模型预测分布  Q  尽可能接近真实分布  P  ，因此用交叉螪$  H(P, Q) $作为损失函数，通过优化模型参数使  H(P, Q)  最小化，本质是让预测分布逼近真实分布。