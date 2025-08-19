---

title: 深度学习（2）如何训练一个神经网络模型
tag: DL DNN
typora-root-url: ./..
---

学会使用PyTorch训练自己的神经网络。

<!--more-->

##### 1.回归问题

构建回归类型的神经网络可以帮助我们解决预测连续值的问题，比如房价预测、温度预测，这里以房价预测为例，搭建一个两层神经网络，有偏置，隐藏层有8个节点，输出层只有一个节点。

###### 1.1 准备数据集

获取数据集的方式有很多，这里记录一下怎么从Kaggle免费下载公开数据集（Kaggle怎么注册参考之前的Kaggle免费使用GPU和TPU笔记）。

首先进入Kaggle官网 https://www.kaggle.com/，在右侧列表找到 Datasets

![](/images/MLPCode/1.png)

在搜索框输入Housing，查找房价数据集，这里下载第一个加州房价数据集（California Housing Dataset）。

![](/images/MLPCode/2.png)

下载后得到一个名为archive的压缩包，里面是加州数据集的CSV格式，这里不解压，将下载的压缩包文件重命名为california_housing，并放在和代码相同的目录下面。

![](/images/MLPCode/3.png)

现在调佣pandas库，读取california_housing压缩包里面的CSV数据集。

~~~
import pandas as pd

# 读取压缩包中的CSV格式的加州数据集
df = pd.read_csv("california_housing.zip", compression='zip')

# 查看数据
print(df)
~~~

如果想知道特征数、特征名、样本数以及数据类型和缺失值信息，可以运行下行代码。

~~~
print(df.info())
~~~

![](/images/MLPCode/5.png)

从上面信息可以看出，这个数据总共有20640个样本，9个特征，median_house_value是房价，也是我们需要预测的值。由于样本总数过大，训练时长过慢，所以随机选择1000个样本。此外，由于特征ocean_proximity类型是字符串，所以这里不考虑该列，直接删除（实际任务中可以换成独热编码）。

~~~
# 随机采样500个样本，删除类别特征ocean_proximity
data = df.sample(n=1000, random_state=42)
# 用dropna()处理缺失值，也可以用fillna()填充数据集
data = data.drop('ocean_proximity', axis=1).dropna() 
~~~

将特征和标签分开，同时前800个样本作为训练数据集，后200个样本作为测试数据集，最后对数据标准化处理，避免量纲不同导致无法收敛。

~~~
# 分离特征和标签
X = data.drop('median_house_value', axis=1).values      # 特征
y = data['median_house_value'].values.reshape(-1, 1)    # 标签（房价）

# 划分训练集（前400）和测试集（后100）
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[-100:], y[-100:]

# 特征标准化
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_std[X_std == 0] = 1e-8  # 避免除零
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# 标签标准化
y_mean, y_std = y_train.mean(), y_train.std()
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std
~~~

将numpy数据转为PyTorch张量。

~~~
import torch

X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test_scaled, dtype=torch.float32)
~~~

###### 1.2 激活函数

使用ReLU激活函数。

~~~
# ReLU激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
~~~

###### 1.3 前向传播

~~~
# 前向传播
def forward(X, W1, b1, W2, b2):
    z1 = torch.matmul(X, W1) + b1
    a1 = relu(z1)
    z2 = torch.matmul(a1, W2) + b2
    return z2
~~~

###### 1.4 损失函数

回归通常使用均方误差。

~~~
# 损失函数 均方误差(MSE)
def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) **2)
~~~

###### 1.5 优化器

使用PyTorch内置的随机梯度下降SGD进行优化。

~~~
# SGD优化器，第一个是参数列表，第二个是学习率
optimizer = torch.optim.SGD(parameters, lr=learning_rate)
~~~

###### 1.6 初始化参数

~~~
# 网络结构：输入层(8特征) → 隐藏层(16神经元) → 输出层(1神经元)
input_dim = X_train.shape[1]
hidden_dim = 16
output_dim = 1

# 初始化权重和偏置（均值0，标准差0.01，需要梯度）
W1 = torch.normal(0, 0.01, size=(input_dim, hidden_dim), requires_grad=True)
b1 = torch.normal(0, 0.01, size=(1, hidden_dim), requires_grad=True)
W2 = torch.normal(0, 0.01, size=(hidden_dim, output_dim), requires_grad=True)
b2 = torch.normal(0, 0.01, size=(1, output_dim), requires_grad=True)

# 超参数，学习率和迭代次数
learning_rate = 0.1
num_epochs = 1000
~~~

###### 1.7 代码汇总，开始训练并使用测试集评估

~~~
import pandas as pd
import torch

# 读取压缩包中的CSV格式的加州数据集
df = pd.read_csv("california_housing.zip", compression='zip')

# 随机采样500个样本，删除类别特征ocean_proximity
data = df.sample(n=1000, random_state=42)
# 用dropna()处理缺失值，也可以用fillna()填充数据集
data = data.drop('ocean_proximity', axis=1).dropna() 

# 分离特征和标签
X = data.drop('median_house_value', axis=1).values      # 特征
y = data['median_house_value'].values.reshape(-1, 1)    # 标签（房价）

# 划分训练集（前400）和测试集（后100）
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[-100:], y[-100:]

# 特征标准化
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_std[X_std == 0] = 1e-8  # 避免除零
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# 标签标准化
y_mean, y_std = y_train.mean(), y_train.std()
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# 将数据转换为PyTorch张量
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test_scaled, dtype=torch.float32)

# ReLU激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
    
# 前向传播
def forward(X, W1, b1, W2, b2):
    z1 = torch.matmul(X, W1) + b1
    a1 = relu(z1)
    z2 = torch.matmul(a1, W2) + b2
    return z2

# 损失函数 均方误差(MSE)
def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) **2)

# 网络结构：输入层(8特征) → 隐藏层(8神经元) → 输出层(1神经元)
input_dim = X_train.shape[1]
hidden_dim = 8
output_dim = 1

# 初始化权重和偏置（均值0，标准差0.01，需要梯度）
W1 = torch.normal(0, 0.01, size=(input_dim, hidden_dim), requires_grad=True)
b1 = torch.normal(0, 0.01, size=(1, hidden_dim), requires_grad=True)
W2 = torch.normal(0, 0.01, size=(hidden_dim, output_dim), requires_grad=True)
b2 = torch.normal(0, 0.01, size=(1, output_dim), requires_grad=True)

# 超参数，学习率和迭代次数
learning_rate = 0.1
num_epochs = 1000

# SGD优化器，第一个是参数列表，第二个是学习率
parameters = [W1, b1, W2, b2] # 参数列表
optimizer = torch.optim.SGD(parameters, lr=learning_rate)

# 记录损失值
train_losses = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = forward(X_train_torch, W1, b1, W2, b2)
    
    # 计算均方误差损失
    loss = mse(y_pred, y_train_torch)
    # 使用.item()将张量转为Python 原生数据，用于绘图
    train_losses.append(loss.item())
    
    # 反向传播 + 参数更新
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    # 每50轮打印一次损失
    if (epoch + 1) % 50 == 0:
        print(f"轮次 {epoch+1}/{num_epochs}, 训练损失: {loss.item():.4f}")

# 在测试集上评估模型
with torch.no_grad():  # 关闭梯度计算，节省内存并加速计算
    y_test_pred = forward(X_test_torch, W1, b1, W2, b2)  #前向传播
    # 计算损失
    test_loss = mse(y_test_torch, y_test_pred) 
    
print(f"\n测试集均方误差（MSE）: {test_loss.item():.4f}")
~~~

###### 1.8 绘制损失函数下降图

~~~
import matplotlib.pyplot as plt

plt.plot(range(1, num_epochs+1), train_losses, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()
~~~

###### 1.9 存在的问题

训练过程中，可能会出损失函数为NaN，或者损失过高等问题。

如果损失过高，可以通过增加隐藏层和隐藏层节点，调高学习率来解决。

如果是NaN问题，先看看数据集有没有标准化，有没有缺失值，此外可能还存在梯度消失等问题，需要调整学习率。

神经网络通常需要依靠调整参数，来获得一个不错的结果。

##### 2.分类问题

在神经网络中，分类通常用来处理图片数据，所以本篇使用Fashion MNIST图像数据集，每个图像是28*28像素。

###### 2.1 准备数据集

同样地，可以使用Kaggle获取数据集，在Datasets输入Fashion MNIST，就能找到。但是，PyTorch可以更方便地加载数据集，所以这里使用PyTorch下载。下载前还需要注意对数据进行标准化处理。

~~~
import torchvision
import torchvision.transforms as transforms
import torch

# 数据转换：将图像转为张量并标准化（像素值从0-255转为0-1）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量 (28x28x1 → 1x28x28)，并自动归一化到[0,1]
])

# 加载训练集
train_dataset = torchvision.datasets.FashionMNIST(
    root='./',            # 数据保存路径
    train=True,           # 训练集
    download=True,        # 若本地没有则自动下载
    transform=transform   # 转为张量并标准化
)

# 加载测试集
test_dataset = torchvision.datasets.FashionMNIST(
    root='./',
    train=False,          # 测试集
    download=True,
    transform=transform
)
~~~

从训练集选择6000个样本用来训练，测试集选1000个样本用来评估。

~~~
# 取前6000个样本
train_dataset = torch.utils.data.Subset(train_dataset, range(6000))

# 取前1000个样本用于评估
test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
~~~

将28*28的像素展开成784维向量，提取数据特征和标签。

~~~
import numpy as np

# 提取数据特征和标签
def prepare_data(dataset):
    images, labels = [], []
    for img, label in dataset:
        # 图像展平：28x28 → 784维向量
        img_flat = img.view(-1).numpy()  # 展平为一维张量后转numpy
        images.append(img_flat)
        labels.append(label)
    # 转换为PyTorch张量
    X = torch.tensor(images, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

# 准备训练集和测试集
X_train, y_train = prepare_data(train_dataset)
X_test, y_test = prepare_data(test_dataset)
~~~

###### 2.2 激活函数

使用ReLU激活函数。

~~~
# ReLU激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
~~~

###### 2.3 前向传播

~~~
# 前向传播
def forward(X, W1, b1, W2, b2):
    z1 = torch.matmul(X, W1) + b1
    a1 = relu(z1)
    z2 = torch.matmul(a1, W2) + b2 
    return z2
~~~

###### 2.4 损失函数

多分类任务使用交叉熵损失（softmax）。

~~~
# 交叉熵损失
def cross_entropy_loss(y_pred, y_true):
    # 计算softmax
    exp_pred = torch.exp(y_pred)
    softmax_pred = exp_pred / torch.sum(exp_pred, dim=1, keepdim=True)
    # 取真实标签对应的概率
    n_samples = y_pred.shape[0]
    log_probs = -torch.log(softmax_pred[range(n_samples), y_true])
    return torch.mean(log_probs)
~~~

###### 2.5 优化器

和回归一样，使用SGD。

~~~
optimizer = torch.optim.SGD(parameters, lr=learning_rate)
~~~

###### 2.6 初始化参数

~~~
# 网络结构：输入层(784=28x28) → 隐藏层(128神经元) → 输出层(10类别)
input_dim = 784        # 28x28图像展平后的维度
hidden_dim = 128       # 隐藏层神经元数
output_dim = 10        # 10个类别（Fashion-MNIST的类别数）

# 初始化权重和偏置（均值0，标准差0.01，需要梯度）
W1 = torch.normal(0, 0.01, size=(input_dim, hidden_dim), requires_grad=True)
b1 = torch.normal(0, 0.01, size=(1, hidden_dim), requires_grad=True)
W2 = torch.normal(0, 0.01, size=(hidden_dim, output_dim), requires_grad=True)
b2 = torch.normal(0, 0.01, size=(1, output_dim), requires_grad=True)

# 超参数
learning_rate = 0.1
num_epochs = 500
~~~

###### 2.7 代码总结，开始训练并测试

~~~
# 分类
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

# 数据转换：将图像转为张量并标准化（像素值从0-255转为0-1）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量 (28x28x1 → 1x28x28)，并自动归一化到[0,1]
])

# 加载训练集
train_dataset = torchvision.datasets.FashionMNIST(
    root='./',            # 数据保存路径
    train=True,           # 训练集
    download=True,        # 若本地没有则自动下载
    transform=transform   # 转为张量并标准化
)

# 加载测试集
test_dataset = torchvision.datasets.FashionMNIST(
    root='./',
    train=False,          # 测试集
    download=True,
    transform=transform
)

# 取前6000个样本
train_dataset = torch.utils.data.Subset(train_dataset, range(6000))

# 取前1000个样本用于评估
test_dataset = torch.utils.data.Subset(test_dataset, range(1000))

# 提取数据特征和标签
def prepare_data(dataset):
    images, labels = [], []
    for img, label in dataset:
        # 图像展平：28x28 → 784维向量
        img_flat = img.view(-1).numpy()  # 展平为一维张量后转numpy
        images.append(img_flat)
        labels.append(label)
    images_np = np.array(images)
    # 转换为PyTorch张量
    X = torch.tensor(images_np, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)  # 分类任务标签用长整数
    return X, y

# 准备训练集和测试集
X_train, y_train = prepare_data(train_dataset)
X_test, y_test = prepare_data(test_dataset)

# ReLU激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 前向传播
def forward(X, W1, b1, W2, b2):
    z1 = torch.matmul(X, W1) + b1
    a1 = relu(z1)
    z2 = torch.matmul(a1, W2) + b2 
    return z2

# 交叉熵损失
def cross_entropy_loss(y_pred, y_true):
    # 计算softmax
    exp_pred = torch.exp(y_pred)
    softmax_pred = exp_pred / torch.sum(exp_pred, dim=1, keepdim=True)
    # 取真实标签对应的概率
    n_samples = y_pred.shape[0]
    log_probs = -torch.log(softmax_pred[range(n_samples), y_true])
    return torch.mean(log_probs)

# 网络结构：输入层(784=28x28) → 隐藏层(128神经元) → 输出层(10类别)
input_dim = 784        # 28x28图像展平后的维度
hidden_dim = 128       # 隐藏层神经元数
output_dim = 10        # 10个类别（Fashion-MNIST的类别数）

# 初始化权重和偏置（均值0，标准差0.01，需要梯度）
W1 = torch.normal(0, 0.01, size=(input_dim, hidden_dim), requires_grad=True)
b1 = torch.normal(0, 0.01, size=(1, hidden_dim), requires_grad=True)
W2 = torch.normal(0, 0.01, size=(hidden_dim, output_dim), requires_grad=True)
b2 = torch.normal(0, 0.01, size=(1, output_dim), requires_grad=True)

# 超参数
learning_rate = 0.1
num_epochs = 500

# 优化器
parameters = [W1, b1, W2, b2]
optimizer = torch.optim.SGD(parameters, lr=learning_rate)

# 记录损失率和准确率
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = forward(X_train, W1, b1, W2, b2)  # 输出未经过softmax的logits
    
    # 计算损失
    loss = cross_entropy_loss(y_pred, y_train)
    train_losses.append(loss.item())
    
    # 计算准确率（预测类别与真实类别对比）
    pred_labels = torch.argmax(y_pred, dim=1)  # 取概率最大的类别
    accuracy = torch.mean((pred_labels == y_train).float())
    train_accuracies.append(accuracy.item())
    
    # 反向传播 + 参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每5轮打印一次
    if (epoch + 1) % 50 == 0:
        print(f"轮次 {epoch+1}/{num_epochs}, 损失: {loss.item():.4f}, 准确率: {accuracy.item():.4f}")

with torch.no_grad():
    y_test_pred = forward(X_test, W1, b1, W2, b2)
    # 测试集损失
    test_loss = cross_entropy_loss(y_test_pred, y_test)
    # 测试集准确率
    test_pred_labels = torch.argmax(y_test_pred, dim=1)
    test_accuracy = torch.mean((test_pred_labels == y_test).float())

print(f"\n测试集损失: {test_loss.item():.4f}")
print(f"测试集准确率: {test_accuracy.item():.4f}")
~~~

###### 2.8 可视化

~~~
import matplotlib.pyplot as plt

# 绘制损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss')
plt.grid(True)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, 'r-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()
~~~

