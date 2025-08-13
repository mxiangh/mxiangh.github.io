---
title: 机器学习（11）朴素贝叶斯——Naive Bayes
tags: ML Classification
---

思想：一种基于概率论的分类算法，在已经知道样本特征的情况下，输出样本对应类别概率。

<!--more-->

##### 1.基本概念

1.1 贝叶斯公式（ X ：特征向量， Y ：类别）：

$$ P(Y \vert X)=\frac{P(X \vert Y) P(Y)}{P(X)} $$

1.2 先验概率：基于以往经验、历史数据或主观判断对事件发生概率的初始估计。

例如：随机抽一张扑克牌是红桃的概率为 13/52=25%。

1.3 后验概率（条件概率）：事情已经发生，要求这件事情发生的原因是由某个因素引起的可能性的大小。

例如：已知抽到的牌是红色，则 “这张牌是红桃” 的概率为13/26=50%（红色牌中红桃占一半）。

$P(A \vert B)$是已知B发生后A的条件概率，也由于得自B的取值而被称作A的后验概率。

$ P(B \vert A) $ 是已知A发生后B的条件概率，也由于得自A的取值而被称作B的后验概率。

1.4 联合概率：表示两件事共同发生（数学概念上的交集）的概率，A和B的联合概率表示为$P(A \cap B)$或$P(A , B)$或$P(AB)$。

1.5 全概率公式：$P(A)=\sum_{i = 1}^{n}P(A \mid B_i)P(B_i)$，把事件 A 发生的概率，通过样本空间的划分，分解为在不同条件 $B_i$ 下 A 发生的概率的加权和，权重就是条件 $B_i$ 本身发生的概率。

1.6 相互独立：对于两个或多个事件，一个事件的发生与否不会改变另一个事件发生的概率，即$P(AB)=P(A) P(B)$。

1.7 朴素：各个特征之间相互独立，那么贝叶斯公式中的 $ \mathrm{P}(\mathrm{X} \mid \mathrm{Y}) $ 可写成：

$$P(X \vert Y)=P\left(x_{1} \vert Y\right) P\left(x_{2} \vert Y\right) \cdots P\left(x_{n} \vert Y\right)$$

朴素贝叶斯公式：

$$P(Y \vert X)=\frac{P\left(x_{1} \vert Y\right) P\left(x_{2} \vert Y\right) \cdots P\left(x_{n} \vert Y\right) P(Y)}{P(X)}$$

##### 2.贝叶斯公式推导

根据条件概率的定义，在事件B发生的条件下事件A发生的概率为：$ P(A \vert B)=\frac{P(A \cap B)}{P(B)} $

同样地，在事件A发生的条件下事件B发生的概率为：$ P(B \vert A)=\frac{P(A \cap B)}{P(A)} $

结合这两个方程式，我们可以得到：$ P(A \vert B) P(B) = P(B \vert A) P(A) $

这个引理有时称为概率乘法规则。上式两边同时除以P(A)，若P(A)是非零的，我们可以得到贝叶斯定理：

$$ P(B \vert A)=\frac{P(A \vert B) P(B)}{P(A)} $$

贝叶斯定理解释：事件X在事件Y发生的条件下的概率，与事件Y在事件X发生的条件下的概率是不一样的，然而这两者是有确定关系的。

贝叶斯公式的用途在于通过已知三个概率来推测第四个概率。

##### 3.朴素贝叶斯定理

3.1 问题描述

现在已经有m个样本，每个样本有n个特征，特征输出有K个类别。给定一个新的样本特征，能否根据已有的样本和标签输出新样本的类别概率？

在样本X特征已知的情况下求Y是类别k的输出概率，这很显然和上述所描述的后验概率$P(Y \vert X)$是同一件事。

根据贝叶斯定理，我们想要求$P(Y \vert X)$，则需要知道$P(X \vert Y)$、$P(Y)$和$P(X)$。

3.2 公式详解

假设我们的分类模型样本是：

$$\left(x_{1}^{(1)}, x_{2}^{(1)}, \ldots x_{n}^{(1)}, y_{1}\right),\left(x_{1}^{(2)}, x_{2}^{(2)}, \ldots x_{n}^{(2)}, y_{2}\right), \ldots\left(x_{1}^{(m)}, x_{2}^{(m)}, \ldots x_{n}^{(m)}, y_{m}\right)$$

即我们有 m 个样本，每个样本有 n 个特征，特征输出有 K 个类别，定义为$ C_{1}, C_{2}, \ldots C_{K}$。

贝叶斯定理用于计算 “新样本属于某类别” 的后验概率，公式为：

$$P(Y = C_k \mid \mathbf{X} = \mathbf{x}) = \frac{P(\mathbf{X} = \mathbf{x} \mid Y = C_k) \cdot P(Y = C_k)}{P(\mathbf{X} = \mathbf{x})}$$

其中：

$Y = C_k$：样本属于第 k 个类别（$k = 1,2,\dots,K$，K 是类别总数 ）；

$\mathbf{X} = \mathbf{x}$：样本的特征向量取$\mathbf{x}$（$\mathbf{x} = (x_1, x_2, \dots, x_n)$，n 是特征维度 ）；

$P(Y = C_k)$：先验概率，类别 $C_k$ 出现的整体概率；

$P(\mathbf{X} = \mathbf{x} \mid Y = C_k)$：似然，类别$C_k$下，特征取$\mathbf{x}$的条件概率；

$P(\mathbf{X} = \mathbf{x})$：特征取 $\mathbf{x}$ 的边缘概率。

（1）类别先验概率

先验概率描述 “类别本身出现的概率”，与特征无关。公式为：

$$ P(Y = C_k) = \frac{\text{训练集中类别 } C_k \text{ 的样本数量}}{\text{训练集总样本数量}} $$

它反映 “在未看特征时，类别 $C_k$ 出现的固有概率”。

（2）似然（类别条件下的特征概率）

似然描述 “已知类别时，特征取特定值的概率”，是贝叶斯公式的核心部分。对特征向量 $\mathbf{X} = (X_1, X_2, \dots, X_n)$，类别 $C_k$ 下特征取 $\mathbf{x} = (x_1, x_2, \dots, x_n)$ 的似然为：

$$P(\mathbf{X} = \mathbf{x} \mid Y = C_k) = P(X_1 = x_1, X_2 = x_2, \dots, X_n = x_n \mid Y = C_k)$$

它反映 “类别 $C_k$ 与特征 $\mathbf{x}$ 的关联紧密程度”。

（3）朴素贝叶斯的简化（条件独立性假设，即样本互不相关）

为了求解更方便，朴素贝叶斯算法引入条件独立性假设：假设特征之间在 “已知类别时相互独立”。此时似然可简化为：

$$P(\mathbf{X} = \mathbf{x} \mid Y = C_k) = \prod_{i=1}^n P(X_i = x_i \mid Y = C_k)$$

即 “类别 $C_k$ 下，特征 $X_i = x_i$ 的概率” 的乘积。这一假设大幅降低了计算复杂度，但会损失一定精度（实际特征可能不独立 ）。

（4）边缘概率

根据全概率公式求解边缘概率：

$$ P(\mathbf{X} = \mathbf{x}) = \sum_{k=1}^K P(\mathbf{X} = \mathbf{x} \vert Y = C_k) \cdot P(Y = C_k) = \sum_{j=1}^K \lbrack \prod_{i=1}^n P(X_i = x_i \mid Y = C_j) \cdot P(Y = C_j) \rbrack $$

（5）计算 “新样本属于某类别”的条件概率

$$P(Y = C_k \mid \mathbf{X} = \mathbf{x}) = \frac{\prod_{i=1}^n P(X_i = x_i \mid Y = C_k) P(Y = C_k)}{\sum_{j=1}^K \lbrack \prod_{i=1}^n P(X_i = x_i \mid Y = C_j) \cdot P(Y = C_j) \rbrack}$$

当知道新样本在各个类别的概率时，只需要输出概率最大的类别就可以了。

##### 4.朴素贝叶斯的变体

4.1 多项式模型（MultinomialNB）

适合离散数据，常用于文本分类。

4.2 高斯模型（GaussianNB）

略

4.3 伯努利模型（BernoulliNB）

略

~~~
~~~



~~~
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 加载西瓜数据集2.0
data = {
    '色泽': ['青绿', '乌黑', '乌黑', '青绿', '浅白', '青绿', '乌黑', '乌黑', 
            '乌黑', '青绿', '浅白', '浅白', '青绿', '浅白', '乌黑', '浅白'],
    '根蒂': ['蜷缩', '蜷缩', '蜷缩', '蜷缩', '蜷缩', '稍蜷', '稍蜷', '稍蜷', 
            '硬挺', '硬挺', '硬挺', '蜷缩', '稍蜷', '稍蜷', '稍蜷', '硬挺'],
    '敲声': ['浊响', '浊响', '浊响', '清脆', '清脆', '浊响', '浊响', '清脆', 
            '清脆', '清脆', '浊响', '浊响', '浊响', '浊响', '清脆', '清脆'],
    '纹理': ['清晰', '清晰', '清晰', '清晰', '模糊', '清晰', '清晰', '清晰', 
            '模糊', '模糊', '模糊', '清晰', '模糊', '清晰', '清晰', '模糊'],
    '脐部': ['凹陷', '凹陷', '凹陷', '凹陷', '平坦', '凹陷', '凹陷', '凹陷', 
            '平坦', '平坦', '平坦', '凹陷', '凹陷', '凹陷', '平坦', '平坦'],
    '触感': ['硬滑', '硬滑', '硬滑', '硬滑', '硬滑', '软粘', '软粘', '硬滑', 
            '软粘', '硬滑', '硬滑', '软粘', '硬滑', '软粘', '硬滑', '硬滑'],
    '好瓜': ['是', '是', '是', '是', '否', '是', '是', '是', '否', '否', '否', 
            '是', '否', '是', '否', '否']
}

# 数据预处理（文本特征编码）
df = pd.DataFrame(data)
le = LabelEncoder()
# 将所有文本特征转换为数值（好瓜：1=是，0=否）
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# 划分特征（X）和标签（y）
X = df.drop('好瓜', axis=1)  # 特征：色泽、根蒂、敲声、纹理、脐部、触感
y = df['好瓜']               # 标签：好瓜（1）/坏瓜（0）
feature_names = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify=y保持类别比例
)

# 初始化并训练朴素贝叶斯模型
clf = GaussianNB()
clf.fit(X_train, y_train)

# 预测与评估
## 测试集预测
y_pred_test = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"测试集准确率：{test_accuracy:.2f}")

~~~

