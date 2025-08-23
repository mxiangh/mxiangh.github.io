---
title: 深度学习（8)Transformer详解
tags: DL Transformer
typora-root-url: ./..
---

一文详解Transformer及Encoder、Decoder、Attention、LayerNorm。

<!--more-->

论文：2017，Attention Is All You Need.

注：这篇blog基于笔者前面所述DNN、CNN、RNN基础上，对Transformer进行详解，对于前面未提过的内容，在本篇会进行介绍。此外，Transformer在刚提出时，是针对NLP任务执行的，所以本篇内容会以NLP中机器翻译举例说明。



