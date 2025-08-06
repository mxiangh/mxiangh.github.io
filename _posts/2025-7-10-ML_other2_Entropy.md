---
title: 机器学习补充（2）KL散度与交叉熵
tags: ML Tips
typora-root-url: ./..
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

#### 4.二分类交叉熵

二分类需要了解Sigmoid函数

![](/assets/images/Algorithm/sigmoid.png)

$$ y = \frac{1}{1+e^{-x}} $$

这个函数将x控制在$\lbrack 0, 1 \rbrack$之间，转为一个接近0或1的y值。

4.1 真实分布P的定义

对于二分类问题，单个样本的真实标签$y \in \lbrace 0, 1 \rbrace $，其真实概率分布P是伯努利分布：

- 当 $ y=1 $时，$P(Y=1)=1, P(Y=0)=0$

- 当 $ y=0 $时，$P(Y=1)=0, P(Y=0)=1$

4.2 模型预测分布Q的定义

模型Sigmoid函数输出预测概率$\hat{y} = Q(Y=1 \vert x)=\frac{1}{1+e^{-x}}$，则预测分布Q为：

- $ Q(Y=1 \vert x) = \hat{y} $

- $ Q(Y=0 \vert x) = 1 - \hat{y} $

4.3 损失函数推导

将P和Q代入交叉熵公式：

$$ H(P, Q) = -\sum_{y \in \lbrace 0,1 \rbrace } P(y) logQ(y \vert x) $$

- 当真实标签y=1时：$ H(P, Q) = -1·log Q(Y=1 \vert x) - 0·log Q(Y=0 \vert x) = - log \hat{y} $

- 当真实标签y=0时：$ H(P, Q) = -0·log Q(Y=1 \vert x) - 1·log Q(Y=0 \vert x) = - log (1- \hat{y}) $

合并上述两种情况，得到单个样本的交叉熵损失函数：

$$ L(\hat{y},y)= - \lbrack y log \hat{y} + (1-y) log (1- \hat{y}) \rbrack $$

对于整个数据集，总损失函数为单个样本损失的平均值，取均值是为了避免求解时梯度爆炸

$$ L(\hat{y},y)= - \frac{1}{m} \sum_{i=1}^m \lbrack y log \hat{y} + (1-y) log (1- \hat{y}) \rbrack $$

#### 5.多分类交叉熵

略