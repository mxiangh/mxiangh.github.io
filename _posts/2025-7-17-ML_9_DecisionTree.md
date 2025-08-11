---
title: 机器学习（9）决策树——Decision Tree¶
tags: ML Regression Classification
---

思想：决策树模型是表示基于特征对实例进行分类或回归的树形结构（二叉树）。决策树可以转换成一个if-then规则的集合，也可以看作是定义在特征空间划分上的类的条件概率分布。

<!--more-->

决策树学习算法包括3部分：特征选择、树的生成和树的剪枝。

##### 1.特征选择

特征选择是指从训练数据中众多的特征中选择一个特征作为当前节点的分裂标准，如何选择特征有着很多不同量化评估标准，从而衍生出不同的决策树算法。

（1）样本集合$D$对特征$A$的信息增益Info-Gain（ID3）


$$g(D, A)=H(D)-H(D \mid A)$$

$$H(D)=-\sum_{k=1}^{K} \frac{\left|C_{k}\right|}{|D|} \log _{2} \frac{\left|C_{k}\right|}{|D|}$$

$$H(D | A)=\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} H\left(D_{i}\right)$$

其中，$H(D)$是数据集$D$的熵，$H(D_i)$是数据集$D_i$的熵，$H(D\mid A)$是数据集$D$对特征$A$的条件熵。	$D_i$是$D$中特征$A$取第$i$个值的样本子集，$C_k$是$D$中属于第$k$类的样本子集。$n$是特征$A$取 值的个数，$K$是类的个数。

（2）样本集合$D$对特征$A$的信息增益比Gain-ration（C4.5）


$$g_{R}(D, A)=\frac{g(D, A)}{H(D)}$$


其中，$g(D,A)$是信息增益，$H(D)$是数据集$D$的熵。

（3）CART（分类与回归树，classification and regression tree）

对于回归树，采用的是平方误差最小化准则；对于分类树，采用基尼指数最小化准则

a.CART回归

假设已将输入空间划分为M个单元$R_1，R_2......R_m$，并且在每个单元$R_m$上有一个固定的输出值$C_m$，于是回归树可以表示为：
$$f(x)=\sum_{m=1}^{M}c_mI(x \in R_m)$$
当输入空间的划分确定时，可以用平方误差来$\sum_{x_i \in R_m}(y_i-f(x_i))^2$表示回归树对于训练数据的预测误差

b.CART分类

基尼指数：假设有K个类，样本点属于第K类的概率为$p_k$，则概率分布的基尼指数定义为

$$\operatorname{Gini}(p)=\sum_{k=1}^{K}p_k(1-p_k)=1-\sum_{k=1}^{K}p_k^2$$

对于给定的样本集合D，基尼指数为

$$\operatorname{Gini}(D)=1-\sum_{k=1}^{K}(\frac{\lvert C_{k}\vert }{\vert D\vert })^{2}$$

如果样本集合D根据特征A是否取某一可能值被分为$D_1$和$D_2$两部分，则特征$A$条件下集合$D$的基尼指数为

 $$\operatorname{Gini}(D, A)=\frac{\vert D_{1}\vert }{\vert D\vert } \operatorname{Gini}\left(D_{1}\right)+\frac{\vert D_{2}\vert }{\vert D\vert } \operatorname{Gini}\left(D_{2}\right)$$

##### 2.树的生成

根据选择的特征评估标准，从上至下递归地生成子节点，直到数据集不可分则停止决策树停止生长。树结构来说，递归结构是最容易理解的方式。

##### 3.树的剪枝

- 预剪枝：边建立决策树边进行剪枝的操作（更实用），预剪枝需要限制深度，叶子节点个数，叶子节点样本数，信息增益量等。
- 后剪枝：当建立完决策树后来进行剪枝操作，通过一定的衡量标准（叶子节点越多，损失越大）

决策树容易过拟合，一般来需要剪枝，缩小树结构规则，缓解过拟合。

~~~
~~~



~~~
# sklearn实现
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 西瓜数据集2.0（周志华西瓜书表4.1）
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

# 创建DataFrame并编码特征
df = pd.DataFrame(data)
# print(df)

le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])  # 将文本转换为数值
# print(df)

# 划分特征和标签
X = df.drop('好瓜', axis=1)  # 特征
y = df['好瓜']               # 标签（好瓜=1，坏瓜=0）
feature_names = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']  # 特征名称
class_names = ['否', '是']  # 标签名称

# 训练决策树（使用信息熵作为分裂准则）
# criterion可选参数：gini（基尼指数）、entropy（信息熵）
# clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# 绘制决策树
plt.figure(figsize=(10, 5))  # 设置图形大小
plot_tree(
    clf,
    feature_names=feature_names,  # 特征名称
    class_names=class_names,      # 类别名称
    filled=True,                  # 节点填充颜色（根据类别区分）
    rounded=True,                 # 节点边框圆角
    fontsize=10,                  # 字体大小
    precision=2                   # 数值精度
)
plt.title('西瓜数据集决策树（基于信息熵）', fontsize=15)
plt.show()  # 显示图形

# 打印特征重要性
print("\n特征重要性（值越高，对分类的贡献越大）：")
for name, importance in zip(feature_names, clf.feature_importances_):
    print(f"{name}: {importance:.4f}")

# samples: 样本数
# value: 样本分类数
# class: 根据第一行条件，判断类别
~~~

