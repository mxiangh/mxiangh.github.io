---

title: 深度学习（5）常用的CNN模型
tag: DL CNN
typora-root-url: ./..
---

学习一些有代表性 的卷积网络架构。

<!--more-->

##### 1.AlexNet

论文：Imagenet classification with deep convolutional neural networks.

![](/images/CNNBlock/1.png)

AlexNet由⼋层组成：五个卷积层、两个全连接隐藏层和⼀个全连接输出层，使⽤ReLU作为激活函数。此外，还使用了暂退法（dropout）。

~~~
import torch
from torch import nn

net_AlexNet = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10))

x_AlexNet = torch.randn(1, 1, 224, 224)
for layer in net_AlexNet:
    x_AlexNet = layer(x_AlexNet)
    print(layer.__class__.__name__,'output shape:\t',x_AlexNet.shape)
~~~

##### 2.VGG

论文：Very deep convolutional networks for large-scale image recognition

VGG常用的有四个模型VGG11(A)、VGG13(B)、VGG16(D)、VGG19(E)，其中，VGG16比较火。VGG通过块的思想构建神经网络，卷积层固定$3 \times 3$、填充1，汇聚层固定$2 \times 2$、步幅2。

每种模型都有五个卷积块（VGG块），每个卷积块内部有数量不一的相同尺寸的卷积核，卷积块后面跟着一个最大汇聚层，五个卷积块后面是三个全连接线性层。

![](/images/CNNBlock/2.png)

![](/images/CNNBlock/3.png)

~~~
import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), 
        nn.ReLU(), 
        nn.Dropout(0.5),
        nn.Linear(4096, 4096), 
        nn.ReLU(), 
        nn.Dropout(0.5),
        nn.Linear(4096, 10))

conv_arch_11 = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
conv_arch_13 = ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512))
conv_arch_16 = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
conv_arch_19 = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))

net_vgg11 = vgg(conv_arch_11)
net_vgg13 = vgg(conv_arch_13)
net_vgg16 = vgg(conv_arch_16)
net_vgg19 = vgg(conv_arch_19)

x_vgg = torch.randn(size=(1, 1, 224, 224))
for blk in net_vgg11:
    x_vgg = blk(x_vgg)
    print(blk.__class__.__name__,'output shape:\t',x_vgg.shape)
~~~

##### 3.NiN

论文：Network in network.

NiN的想法是在每个像素位置应⽤⼀个全连接层。如果我们将权重连接到每个空间位置，我们可以将其视为1 × 1卷积层，或作为在每个像素位置上独⽴作⽤的全连接层。从另⼀个⻆度看，即将空间维度中的每个像素视为单个样本，将通道维度视为不同特征（feature）。

NiN块以⼀个普通卷积层开始，后⾯是两个$1 × 1$的卷积层。这两个$1 × 1$卷积层充当带有ReLU激活函数的逐像素全连接层。第⼀层的卷积窗⼝形状通常由用户设置。随后的卷积窗⼝形状固定为$1 × 1$。

![](/images/CNNBlock/4.png)

NiN完全取消了全连接层，其输出通道数等于标签类别的数量。最后放⼀个全局平均汇聚层（global average pooling layer），⽣成⼀个对数⼏率（logits）。

~~~
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), 
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), 
        nn.ReLU())

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    # 自适应平均池化层将图像尺寸变为1*1
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成⼆维的输出，其形状为(批量⼤⼩,10)
    nn.Flatten())

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
~~~

##### 4.GoogLeNet

论文：Going deeper with convolutions. 

GoogLeNet也被成为**Inception V1**，在此之后，还有**Inception V2**、**Inception V3**、**Inception V4**、**Inception-ResNet V2**。

Inception块由四条并⾏路径组成。前三条路径使⽤窗⼝⼤⼩为1 × 1、3 × 3和5 × 5的卷积层，从不同空间⼤⼩中提取信息。中间的两条路径在输⼊上执⾏1 × 1卷积，以减少通道数，从⽽降低模型的复杂性。第四条路径使⽤3 × 3最⼤汇聚层，然后使⽤1 × 1卷积层来改变通道数。这四条路径都使⽤合适的填充来使输⼊与输出的⾼和宽⼀致，最后我们将每条线路的输出在通道维度上拼接，并构成Inception块的输出。在Inception块中，通常调整的超参数是每层输出通道数。

![](/images/CNNBlock/5.png)

GoogLeNet⼀共使⽤9个Inception块和全局平均汇聚层的堆叠来⽣成其估计值。Inception块之间的最⼤汇聚层可降低维度。第⼀个模块类似于AlexNet和LeNet，Inception块的组合从VGG继承，全局平均汇聚层避免了在最后使⽤全连接层。

![](/images/CNNBlock/6.png)

GoogLeNet模型的内部设计比较复杂。

~~~
# Inception V1
import torch
from torch import nn
from torch.nn import functional as F

# Inception块
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最⼤汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上拼接输出
        return torch.cat((p1, p2, p3, p4), dim=1)

# 第一层卷积 + 最大汇聚层
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第二、三层卷积 + 最大汇聚层
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 两个串联的Inception块 + 最大汇聚层
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                    Inception(256, 128, (128, 192), (32, 96), 64),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 四个串联的Inception块 + 最大汇聚层
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                    Inception(512, 160, (112, 224), (24, 64), 64),
                    Inception(512, 128, (128, 256), (24, 64), 64),
                    Inception(512, 112, (144, 288), (32, 64), 64),
                    Inception(528, 256, (160, 320), (32, 128), 128),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 两个串联的Inception块 + 自适应平均池化层
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                    Inception(832, 384, (192, 384), (48, 128), 128),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
~~~

这里只给出了Inception V1的模型构建，之后的几个Inception版本并没有给出，如果在之后，要做模型对比，即相同数据集下准确率的比较，可能会用到这些网络。

##### 5.批量规范化层

简单介绍一下批量规范化（batch normalization）。

批量规范化是指对于一小批量的特征进行标准化，使其平均值为0，方差为1。如果我们尝试使⽤⼤⼩为1的⼩批量应⽤批量规范化，我们将⽆法学到任何东西。这是因为在减去均值之后，每个隐藏单元将为0。所以，只有使⽤⾜够⼤的⼩批量，批量规范化这种⽅法才是有效且稳定的。

~~~
# 参数为特征数
nn.BatchNorm2d(16)
~~~

这个方法可以提高模型训练的准确率，但是其底层原因是为什么，值得深思。

##### 6.残差网络ResNet

论文：Deep Residual Learning for Image Recognition.

残差网络版本如下：ResNet18、ResNet34、ResNet50、ResNet101、ResNet152。数字表示网络的层数，与Inception一样，可以用来做模型对比。此外，由于这个模型涉及的知识比较复杂，下面进行详细介绍。

随着CNN的发展和普及，人们发现增加神经网络的层数可以提高训练精度，但是如果只是单纯的增加网络的深度，可能会出现“梯度弥散”和“梯度爆炸”等问题。

- 梯度消失：靠近输入层的梯度几乎为零，参数无法更新。

- 梯度爆炸：靠近输入层的梯度很大，参数变得异常大。

传统的解决方法则是权重的初始化(normalized initializatiton)和批量规范化(batch normlization)，虽然解决了梯度问题，但是深度加深带来了另外的问题，就是网络性能的**退化现象**，可以简单的理解为，随着训练轮数（epoch）的增加，精度到达一定程度后，就开始下降了。
![](/images/CNNBlock/7.png)

由上图可以看出，56层的神经网络比22层的神经网络在训练集和测试集的准确率都要差得多，也就是所说的退化现象，也就是说如果只是简单的增加网络的深度，可能会导致神经网络模型退化，进而丢失网络前面获取的特征。

想让深度神经网络越复杂越好用，得让新模型包含旧模型的能力（嵌套）。加层时，若能把新层训练成 “恒等映射”，新模型至少不会更差，还可能更好，所以加层通常能让训练误差更低。

- 恒等映射：对任意集合$A$，如果映射 $f : A→A $定义为 $f(a)=a$，即规定 $A $中每个元素$ a $与自身对应，则称$ f $为 $A $上的恒等映射(identical mapping)。



###### 7.稠密连接网络DenseNet
