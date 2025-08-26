---
title: 深度学习（8)Transformer详解
tags: DL Transformer
typora-root-url: ./..
---

一文详解Transformer及Encoder、Decoder、Attention、LayerNorm。

<!--more-->

论文：2017，Attention Is All You Need.

注：这篇blog基于笔者前面所述DNN、CNN、RNN基础上，对Transformer进行详解，对于前面未提过的内容，在本篇会进行介绍。此外，Transformer在刚提出时，是针对NLP任务执行的，所以本篇内容会以NLP中机器翻译举例说明。

##### 1.编码器Encoder

Encoder负责将输入序列压缩成指定长度的向量，即把外界的各种信息转成计算机能看懂的数据。

![](/images/transformer/1.png)

##### 2.解码器Decoder

Decoder负责把编码器输出的向量进行解码，即把数据转成人能看懂的信息。

![](/images/transformer/2.png)



##### 3.注意力机制

