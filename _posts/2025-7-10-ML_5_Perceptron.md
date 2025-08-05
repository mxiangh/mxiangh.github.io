---
title: 机器学习（5）感知机——Perceptron
tags: ML Classification
typora-root-url: ./..
---

#### 感知机——Perceptron

思想：寻找一个超平面，将线性可分的数据集划分成两类。

<!--more-->

##### 1.超平面

超平面H是从n维空间到n-1维空间的一个映射子空间，它有一个 n 维向量和一个实数定义。因为是子空间，所以超平面一定过原点。

通俗来说，只需要知道超平面在二维是一条直线，在三维是一个平面，依次类推即可。

##### 2.感知机

定义：假设输入空间（特征空间）是 $ \mathcal{X} \subseteq R^{n} $ ，输出空间是 $ \mathcal{Y}=\{+1,-1\} $ 。输入 $ x \in \mathcal{X} $ 表示实例的特征向量，对应于输入空间（特征空间）的点；输出 $ y \in \mathcal{Y} $ 表示实例的类别。由输入空间到输出空间的如下函数

$$f(x)=\operatorname{sign}(w \cdot x+b)$$

称为感知机。

其中：
- $ w $ 和 $ b $ 为感知机模型参数（超平面参数），$ w \in R^{n} $ 叫作权值，$ b \in R $ 叫作偏置；
- $ w \cdot x $ 表示 $ w $ 和 $ x $ 的内积；
- sign 是符号函数，即
$$ \operatorname{sign}(x)=\left\{\begin{array}{ll}+1, & x \geq 0, \\ -1, & x<0 .\end{array}\right. $$
- 感知机对应的超平面 $ w x+b=0 $ 称为分离超平面；

说简单点就是，用一个超平面将两类点分开，因为只有-1和+1，对应了不同的两个类别。

![](/assets/images/Perceptron-Linear-Algorithm/one.png)

如果在二维空间上，两类不同的点能被一条直线完全分开，则称为线性可分，使用感知机的前提是数据一定线性可分。

严格数学定义：

设$D_{0}$和$D_{1}$是 $\mathrm{n}$ 维欧氏空间中的两个点集，如果存在$\mathrm{n}$维向量$\mathrm{w}$和实数$\mathrm{b}$, 使得：

- 所有属于$D_{0}$的点$x_{i}$都有$w x_{i}+b>0$

- 而对于所有属于$D_{1}$的点$x_{j}$则有 $w x_{j}+b<0$， 则我们称 $D_{0}$和$D_{1}$线性可分

- 从二维扩展到多维空间中时, 将$D_{0}$和$ D_{1}$完全正确地划分开的 $w x+b=0$就成了一个超平面。

##### 3.损失函数

假设数据集线性可分，注意到下列现象：

- 正确分类时，当w$x+b\ge 0$是正类，则$y=1$，有$y(wx+b) \ge 0$
- 正确分类时，当w$x+b\le 0$是负类，则$y=-1$，也有$y(wx+b) \ge 0$
- 那么误分类的数据一定有$-y(wx+b) > 0$

设误分类的点集为M，则定义损失函数如下：

$$ L(w,b) = -\sum_{x_i \in M}y_i(w·x_i+b)$$

1. 这个损失函数是非负的；

2. 如果没有误分类点，则损失函数为0。

我们希望误分类点越少越好，很容易想到可以最小化这个损失函数，让函数逼近0。

于是得到待优化的目标函数：$\underset{w,b}{min} L(w,b)$

通过求解参数w和b找到一个超平面将两类数据完全分类。

由于感知机目标函数不可导，无法直接求解析解，所以只能通过迭代优化求取参数。通常使用随机梯度下降法（SGD）优化算法。

注：[随机梯度下降SGD](https://mxiangh.github.io/2025-7-10-ML_other1_Gradient.md)

##### 4.缺点

感知机是一个线性模型，最大的缺点是不能表示异或函数，这为后续神经网络中激活函数的出现做铺垫。

考虑只有两个变量的情况，异或⊕运算规则如下：

| $x_1$ | $x_2$ | $x_1⊕x_2$ |
| ----- | ----- | --------- |
| 0     | 0     | 0         |
| 0     | 1     | 1         |
| 1     | 0     | 1         |
| 1     | 1     | 0         |

用图绘制异或函数的四个点，很明显一条直线分不开

![](/assets/images/Perceptron-Linear-Algorithm/two.png)

为了严谨给出严格的数学证明。考虑如下的感知机模型：

$$f(x)=sign\left(w^{T} x+b\right)$$


其中  $x=\left(x_{1}, x_{2}\right)^{T}, w=\left(w_{1}, w_{2}\right)^{T}, ~ \operatorname{sign}(x)$是符号函数 ．接下来我们证明感知机不能表示异或。

反证法．假设感知机可以模拟异或运算，则必须满足：
- 当 $x=(0,0)^{T}  $时，有 $ f(x)=0 $ ，从而 $ b<0  $；
- 当 $ x=(1,0)^{T}  $时，有 $ f(x)=1 $ ，从而 $ w_{1}>-b>0 $ ；$
- 当 $ x=(0,1)^{T}  $时，有 $ f(x)=1 $ ，从而 $ w_{2}>-b>0 $ ；$
- 但是，当  $x=(1,1)^{T}  $时，有： $f(x)=\operatorname{sign}\left(w_{1}+w_{2}+b\right)=1  $，与  $x_{1} \oplus x_{2}=0  $矛盾。

因此，原假设不成立，感知机无法模拟异或逻辑运算。

~~~
# 手写实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=100):
        self.lr = learning_rate      # 学习率
        self.max_epochs = max_epochs # 最大迭代次数
        self.weights = None          # 权重
        self.bias = None             # 偏置项

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 将标签转换为 ±1（感知机要求）
        y_ = np.where(y == 0, -1, 1)

        for _ in range(self.max_epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) <= 0
                if condition:
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

# 加载鸢尾花数据集（二分类：Setosa vs Versicolor）
iris = load_iris()
X = iris.data[:100, :2]  # 只取前两列特征（方便可视化）
y = iris.target[:100]    # 只取前两类（0和1）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练感知机
perceptron = Perceptron(learning_rate=0.1, max_epochs=75)
perceptron.fit(X_train, y_train)

# 预测并评估
y_pred = perceptron.predict(X_test)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.2f}")
y_train_pred = perceptron.predict(X_train)
print("训练集准确率:", accuracy_score(y_train, y_train_pred))

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
    plt.title('Perceptron Decision Boundary')
    
plot_decision_boundary(X_train, y_train, perceptron)
plt.show()
~~~

