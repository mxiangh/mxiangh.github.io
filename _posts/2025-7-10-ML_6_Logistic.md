---
title: 机器学习（6）逻辑斯蒂回归——Logistic回归
tags: ML Classification
---

#### 逻辑斯蒂回归——Logistic回归

思想：使用线性模型实现分类任务。

<!--more-->

##### 1.线性回归延伸

当我们希望线性模型的预测值逼近真实标记у时，就得到了线性回归模型：

$$ y = w^T x + b $$

那如果将y更换成$ln y$，就可以让线性模型逼近一个指数，即

$$ ln y = w^T x + b $$

这就是对数线性回归，实际上是在试图让$ e^{w^T x + b}$逼近y。这个式子在形式上仍然是线性回归，但实质上已是在求取输入空间到输出空间的非线性函数映射。

这里的对数函数起到了将线性回归模型的预测值（$ w^T x + b$）与真实标记（y）联系起来的作用。

##### 2.线性回归与分类任务的联系

那将线性回归模型运用到分类任务中，只需找一个单调可微函数将分类任务的真实标记y与线性回归模型的预测值联系起来。

对于一个二分类问题，真实输出标记$y \in \lbrace 0, 1 \rbrace $，而线性回归模型$ z = w^T x + b $产生的是一个实际的数值z，于是需要将实值z转换为0/1值。

最理想的是“单位阶跃函数”

$$ y=\left\{\begin{array}{1}
0,z<0\\
0.5,z=0 \\
1,z>0
\end{array}\right. $$

如果预测值z大于0就判为正例，小于0就判为负例，为临界值则任意判别。

但是，单位阶跃函数不连续，于是提出一个常用的替代函数，“Sigmoid函数”

$$ y = \frac{1}{1+e^{-z}} $$

这个函数将z控制在$\lbrack 0, 1 \rbrack$之间，转为一个接近0或1的y值，代入线性模型，得到

$$ y = \frac{1}{1+e^{-(w^T x + b)}} $$

或者变为

$$ ln \frac{y}{1-y} = w^T x + b$$

##### 3.Logistic回归

如果把y看做样本x是正例的概率，则1-y是其反例概率（非正即反，概率为1），两者的比值是

$$ \frac{y}{1-y} $$

这个式子称为几率，表示 “正例发生的概率与反例发生的概率的比值” ，反映正例比反例更可能发生或更不可能发生的程度。

对几率取对数则得到对数几率

$$ ln \frac{y}{1-y} $$

也就是说，前面提到的$ ln \frac{y}{1-y} = w^T x + b$，实际上是在用线性回归模型的预测结果去逼近真是标记的对数几率，而这个模型就是Logistic回归。

虽然名字含有回归，但实际上是用回归模型解决分类问题，并且它输出的是某个类别的近似概率预测。

##### 4.损失函数

获得Logistic回归的损失函数有两种方法，一种是使用极大似然估计，另一种是交叉熵。

4.1 极大似然估计

略

4.2 交叉熵损失函数

 [交叉熵](https://mxiangh.github.io/2025/07/10/ML_other2_Entropy.html) 

4.2.1 真实分布P的定义

对于二分类问题，单个样本的真实标签$y \in \lbrace 0, 1 \rbrace $，其真实概率分布P是伯努利分布：

- 当 $ y=1 $时，$P(Y=1)=1, P(Y=0)=0$

- 当 $ y=0 $时，$P(Y=1)=0, P(Y=0)=1$

4.2.2 模型预测分布Q的定义

模型通过Logistic回归输出预测概率$\hat{y} = Q(Y=1 \vert x)=\frac{1}{1+e^{-(w^T x + b)}}$，则预测分布Q为：

- $ Q(Y=1 \vert x) = \hat{y} $

- $ Q(Y=0 \vert x) = 1 - \hat{y} $

4.2.3 损失函数推导

将P和Q代入交叉熵公式：

$$ H(P, Q) = -\sum_{y \in \lbrace 0,1 \rbrace } P(y) logQ(y \vert x) $$

- 当真实标签y=1时：$ H(P, Q) = -1·log Q(Y=1 \vert x) - 0·log Q(Y=0 \vert x) = - log \hat{y} $

- 当真实标签y=0时：$ H(P, Q) = -0·log Q(Y=1 \vert x) - 1·log Q(Y=0 \vert x) = - log (1- \hat{y}) $

合并上述两种情况，得到单个样本的交叉熵损失函数：

$$ L(\hat{y},y)= - \lbrack y log \hat{y} + (1-y) log (1- \hat{y}) \rbrack $$

对于整个数据集，总损失函数为单个样本损失的平均值，取均值是为了避免求解时梯度爆炸

$$ J(w, b)= - \frac{1}{m} \sum_{i=1}^m \lbrack y log \hat{y} + (1-y) log (1- \hat{y}) \rbrack $$

##### 5.梯度下降求解参数w和b

5.1 w和b梯度，数学推导

5.1.1 函数整理

损失函数：$ J(w, b)= - \frac{1}{m} \sum_{i=1}^m \lbrack y_i log \hat{y}_i + (1-y_i) log (1- \hat{y}_i) \rbrack $

预测函数：$ \hat{y}_i = \sigma (z_i) = \frac{1}{1+e^{z_i}} $

线性函数：$ z_i = -(w^T x_i + b) = \sum_{j=1}^n w_j x_{ij} + b $

5.1.2 b的梯度

利用链式求导法则求解单个样本：

$$ \frac{\partial J}{\partial b} = \frac{\partial J}{\partial \hat{y}_{i}} \frac{\partial \hat{y}_{i}}{\partial z_{i}} \frac{\partial z_{i}}{\partial b}  $$

$$ \frac{\partial J}{\partial \hat{y}_{i}} = - \frac{y_i}{\hat{y}_{i}} + \frac{1 - y_i}{1 - \hat{y}_{i}} = \frac{\hat{y}_{i} - y_i}{\hat{y}_{i}(1-\hat{y}_{i})} $$

$$ \frac{\partial \hat{y}_{i}}{\partial z_{i}} = \hat{y}_{i}(1-\hat{y}_{i}) $$

$$ \frac{\partial z_{i}}{\partial b} = 1$$

合并上述式子

$$ \frac{\partial J}{\partial b} = \frac{\hat{y}_{i} - y_i}{\hat{y}_{i}(1-\hat{y}_{i})}·\hat{y}_{i}(1-\hat{y}_{i})·1 = \hat{y}_{i} -y_{i} $$

总体求和得到b的梯度

$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_{i} -y_{i}) $$

5.1.3 $w_j$的梯度

利用链式求导法则求解单个样本:

$$ \frac{\partial J}{\partial w_j} = \frac{\partial J}{\partial \hat{y}_{i}} \frac{\partial \hat{y}_{i}}{\partial z_{i}} \frac{\partial z_{i}}{\partial w_j}  $$

$$ \frac{\partial J}{\partial \hat{y}_{i}} = - \frac{y_i}{\hat{y}_{i}} + \frac{1 - y_i}{1 - \hat{y}_{i}} = \frac{\hat{y}_{i} - y_i}{\hat{y}_{i}(1-\hat{y}_{i})} $$

$$ \frac{\partial \hat{y}_{i}}{\partial z_{i}} = \hat{y}_{i}(1-\hat{y}_{i}) $$

$$ \frac{\partial z_{i}}{\partial w_j} = x_{ij}$$

合并上述式子：

$$ \frac{\partial J}{\partial w_j} = \frac{\hat{y}_{i} - y_i}{\hat{y}_{i}(1-\hat{y}_{i})} · \hat{y}_{i}(1-\hat{y}_{i}) · x_{ij} = (\hat{y}_{i} -y_{i})x_{ij} $$

总体求和得到$w_j$的梯度

$$\frac{\partial J}{\partial w_j} =\frac{1}{m} \sum_{i=1}^{m} (\hat{y}_{i} -y_{i})x_{ij}$$

5.2 梯度下降求解步骤

注：[随机梯度下降SGD](https://mxiangh.github.io/2025/07/10/ML_other1_Gradient.html)

- 初始化参数：随机或零初始化权重w和偏置b。
- 计算梯度：求损失函数对w和b的偏导数（梯度）：
    - 对偏置b的梯度：$\frac{\partial J}{\partial b} =\frac{1}{m} \sum_{i=1}^{m} (\hat{y}_{i} -y_{i})$
    - 对权重$w_j$的梯度：$\frac{\partial J}{\partial w_j} =\frac{1}{m} \sum_{i=1}^{m} (\hat{y}_{i} -y_{i})x_{ij}$
- 更新参数：沿梯度负方向更新参数（学习率$\alpha$控制步长）：
    - $ w_j = w_j - \alpha \frac{\partial J}{\partial w_j} $
    - $ b = b - \alpha \frac{\partial J}{\partial b} $
- 迭代收敛：重复步骤 2-3，直到损失函数变化小于阈值或达到最大迭代次数。

~~~
# 手写实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.lr = learning_rate      # 学习率
        self.max_epochs = max_epochs # 最大迭代次数
        self.weights = None          # 权重
        self.bias = None             # 偏置项

    # Sigmoid激活函数，将线性输出映射到[0,1]
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降优化
        for _ in range(self.max_epochs):
            # 计算线性输出和预测概率
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred_proba = self.sigmoid(linear_output)

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred_proba - y))
            db = (1 / n_samples) * np.sum(y_pred_proba - y)

            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred_proba = self.sigmoid(linear_output)
        # 概率大于0.5则预测为1，否则为0
        return np.where(y_pred_proba >= 0.5, 1, 0)

# 加载鸢尾花数据集（二分类：Setosa vs Versicolor）
iris = load_iris()
X = iris.data[:100, :2]  # 只取前两列特征（方便可视化）
y = iris.target[:100]    # 只取前两类（0和1）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练Logistic回归模型
log_reg = LogisticRegression(learning_rate=0.1, max_epochs=1000)
log_reg.fit(X_train, y_train)

# 预测并评估
y_pred = log_reg.predict(X_test)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.2f}")
y_train_pred = log_reg.predict(X_train)
print(f"训练集准确率: {accuracy_score(y_train, y_train_pred):.2f}")

# 可视化决策边界
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Logistic Regression Decision Boundary')
    
plot_decision_boundary(X_train, y_train, log_reg)
plt.show()
~~~



~~~
~~~

