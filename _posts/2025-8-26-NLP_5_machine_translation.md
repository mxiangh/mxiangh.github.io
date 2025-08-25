---
title: 自然语言处理（2）机器翻译案例：英—法
tags: DL RNN NLP
typora-root-url: ./..
---

实现自然语言处理中的机器翻译案例。

<!--more-->

##### 1.机器翻译与seq2seq

机器翻译（machine translation）指的是将序列从⼀种语言自动翻译成另⼀种语言。机器翻译有两大流派，一个是统计机器翻译（statistical machine translation），另一个是神经机器翻译（neural machine translation），本篇及之后都以神经机器翻译为主。

机器翻译中的输入序列和输出序列都是长度可变的。为了解决这类问题，本篇使用seq2seq模型，在最开始的seq2seq模型中，使用的是两个LSTM，这里使用其他的循环神经网络 构造一个“编码器—解码器”架构。

##### 2.数据预处理

###### 2.1 数据集下载

数据集下载网址：http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip

下载后，用zipfile库函数导入压缩包里面的数据。

~~~
# 获取数据
import zipfile

# 压缩包路径（根据实际文件位置修改）
zip_path = "fra-eng.zip"
# 压缩包内目标文件的路径（注意目录结构是否正确）
file_in_zip = "fra-eng/fra.txt"

def read_data_nmt(zip_path, file_in_zip):
    # 打开压缩包并读取文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 检查目标文件是否存在于压缩包中
        if file_in_zip in zip_ref.namelist():
            # 读取文件内容（按utf-8编码解码）
            with zip_ref.open(file_in_zip, 'r') as file:
                return file.read().decode('utf-8')
        else:
            print(f"错误：压缩包中未找到 {file_in_zip} 文件")

raw_text = read_data_nmt(zip_path, file_in_zip)

# 打印前100个字符查看内容
print("文件内容预览：")
print(raw_text[:100])
~~~

###### 2.2 数据预处理

~~~
# 数据预处理
def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使⽤空格替换不间断空格
    # 使⽤⼩写字⺟替换⼤写字⺟
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插⼊空格，防止计算机把单词+标点识别成一个整体
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)
    
text = preprocess_nmt(raw_text)
print(text[:80])
~~~

###### 2.3 词元化

与上一节的语言模型文本预测中的字符词元化不一样，在机器翻译中，通常对单词词元化。

下面的tokenize_nmt函数对前num_examples个文本序列对进行词元，其中每个词元要么是一个词，要么是一个标点符号。此函数返回两个词元列表：source和target：source[i]是源语言（这里是英语）第i个文本序列的词元列表，target[i门]是目标语言（这里是法语）第i个文本序列的词元列表。

~~~
# 词元化“英语－法语”数据数据集
# num_examples：最大处理的样本数量
def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    # '\n'为换行符，将文本按换行符拆分
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        # '\t'为制表符，将源和目标拆分
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
~~~

###### 2.4 词表

由于机器翻译数据集由语言对组成，因此我们可以分别为源语言和目标语言构建两个词表。

使用单词级词元化时，词表大小将明显大于使用字符级词元化时的词表大小。为了缓解这一问题，这里我们将出现次数少于2次的低频率词元视为相同的未知（“\<unk\>”）词元。

除此之外，我们还指定了额外的特定词元，例如在小批量时用于将序列填充到相同长度的填充词元（“\<pad\>”），以及序列的开始词元（“\<bos\>”）和结束词元（“\<eos\>”）。这些特殊词元在自然语言处理任务中比较常用。

~~~
import collections

class Vocab:
    """⽂本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
                
    def __len__(self):
        return len(self.idx_to_token)
        
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
        
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
        
    @property
    def unk(self): # 未知词元的索引为0
        return 0
        
    @property
    def token_freqs(self):
        return self._token_freqs
            
def count_corpus(tokens):
    # 统计词元的频率
    # 这⾥的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成⼀个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)

# 可以通过下列方式查找索引对应词元
for idx in range(10):
    j += 1
    if j == 10:
        break
    token = src_vocab.to_tokens(idx)  # 通过索引查词元
    print(f"索引 {idx} 对应的词元：{token}")
~~~

###### 2.5 加载数据集

语言模型中，每个序列样本的长度由时间步num_steps固定，但是在机器翻译中，每个样本是由源和目标组成的文本序列对，每个文本序列可能具有不同长度。为了提高计算效率，仍然可以通过截断（truncation）和填充（padding）方式实现一次只处理一个小批量的文本序列。

假设同一个小批量中的每个序列都应该具有相同的长度num_steps，那么如果文本序列的词元数目少于num_steps时，我们将继续在其末尾添加特定的“\<pad\>”词元，直到其长度达到num_steps；

反之，我们将截断文本序列时，只取其前num_steps个词元，并且丢弃剩余的词元。这样，每个文本序列将具有相同的长度，以便以相同形状的小批量进行加载。

~~~
# 截断或填充⽂本序列
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line)) # 填充

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
~~~

现在我们定义一个函数，可以将文本序列转换成小批量数据集用于训练。我们将特定的“\<eos\>”词元添加到所有序列的末尾，用于表示序列的结束。当模型通过一个词元接一个词元地生成序列进行预测时，生成的“\<eos\>”词元说明完成了序列输出工作。此外，我们还记录了每个文本序列的长度，统计长度时排除了填充词元。

~~~
# 将机器翻译的文本序列转换成⼩批量
def build_array_nmt(lines, vocab, num_steps):
    # 将原文本用词表按行映射到数字
    lines = [vocab[l] for l in lines]
    # 给每行添加结束词元
    lines = [l + [vocab['<eos>']] for l in lines]
    # 截断或填充每行文本，固定长度为时间步num_steps
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    # 记录每个文本序列长度
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len
~~~

###### 2.6 训练模型

最后，我们定义load_data_nmt函数来返回数据迭代器，以及源语言和目标语言的两种词表。

~~~
import torch
import torch.utils.data as data

# 构造⼀个PyTorch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 返回翻译数据集的迭代器和词表
def load_data_nmt(batch_size, num_steps, zip_path, file_in_zip, num_examples=600):
    # 数据预处理
    text = preprocess_nmt(read_data_nmt(zip_path, file_in_zip))
    # 词元化后的数据集
    source, target = tokenize_nmt(text, num_examples)
    # 源词表（英语词表）
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 目标词表（法语词表）
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 源词表转为小批量
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    # 目标词表转为小批量
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 源小批量，小批量长度，目标小批量，目标长度
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    # 构造迭代器
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
~~~

下面我们读出“英语一法语”数据集中的第一个小批量数据。

~~~
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, zip_path=zip_path, file_in_zip=file_in_zip, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效⻓度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效⻓度:', Y_valid_len)
    break
~~~

###### 2.7 代码汇总

~~~
# 代码汇总
import zipfile
import collections
import torch
import torch.utils.data as data

def read_data_nmt(zip_path, file_in_zip):
    # 打开压缩包并读取文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 检查目标文件是否存在于压缩包中
        if file_in_zip in zip_ref.namelist():
            # 读取文件内容（按utf-8编码解码）
            with zip_ref.open(file_in_zip, 'r') as file:
                return file.read().decode('utf-8')
        else:
            print(f"错误：压缩包中未找到 {file_in_zip} 文件")

# 数据预处理
def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使⽤空格替换不间断空格
    # 使⽤⼩写字⺟替换⼤写字⺟
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插⼊空格，防止计算机把单词+标点识别成一个整体
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)
    
# 词元化“英语－法语”数据数据集
# num_examples：最大处理的样本数量
def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    # '\n'为换行符，将文本按换行符拆分
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        # '\t'为制表符，将源和目标拆分
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

class Vocab:
    """⽂本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
                
    def __len__(self):
        return len(self.idx_to_token)
        
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
        
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
        
    @property
    def unk(self): # 未知词元的索引为0
        return 0
        
    @property
    def token_freqs(self):
        return self._token_freqs
            
def count_corpus(tokens):
    # 统计词元的频率
    # 这⾥的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成⼀个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 截断或填充⽂本序列
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line)) # 填充

# 将机器翻译的文本序列转换成⼩批量
def build_array_nmt(lines, vocab, num_steps):
    # 将原文本用词表按行映射到数字
    lines = [vocab[l] for l in lines]
    # 给每行添加结束词元
    lines = [l + [vocab['<eos>']] for l in lines]
    # 截断或填充每行文本，固定长度为时间步num_steps
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    # 记录每个文本序列长度
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

# 数据迭代器，每次返回batch_size个样本
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 返回翻译数据集的迭代器和词表
def load_data_nmt(batch_size, num_steps, zip_path, file_in_zip, num_examples=600):
    # 数据预处理
    text = preprocess_nmt(read_data_nmt(zip_path, file_in_zip))
    # 词元化后的数据集
    source, target = tokenize_nmt(text, num_examples)
    # 源词表（英语词表）
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 目标词表（法语词表）
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 源词表转为小批量
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    # 目标词表转为小批量
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 源小批量，小批量长度，目标小批量，目标长度
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    # 构造迭代器
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

# 压缩包路径（根据实际文件位置修改）
zip_path = "fra-eng.zip"
# 压缩包内目标文件的路径（注意目录结构是否正确）
file_in_zip = "fra-eng/fra.txt"

# train_iter：数据迭代器（源小批量，小批量长度，目标小批量，目标长度）
# src_vocab：源词表，tgt_vocab：目标词表
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, zip_path=zip_path, file_in_zip=file_in_zip, num_steps=8)

# 打印一次迭代器内容
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效⻓度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效⻓度:', Y_valid_len)
    break
~~~

##### 3.编码器—解码器架构

遵循编码器一解码器架构的设计原则，循环神经网络编码器使用长度可变的序列作为输入，将其转换为固定形状的隐状态。换言之，输入序列的信息被编码到循环神经网络编码器的隐状态中。

为了连续生成输出序列的词元，独立的循环神经网络解码器是基于输入序列的编码信息和输出序列已经看见的或者生成的词元来预测下一个词元。

![](/images/NLP/12.png)

图中，\<eos\>表示序列结束词元，一旦输出序列生成此词元，模型就会停止预测。\<bos\>表示序列开始词元，是解码器的输入序列的第一个词元。其次，使用循环神经网络编码器最终的隐状态来初始化解码器的隐状态。

注：编码器最终的隐状态可以看作是输入序列的编码信息，用这个编码信息初始解码器隐状态，等于将输入序列信息的上下文向量传递给解码器。

###### 3.1 编码器

编码器将长度可变的输入序列转换成形状固定的上下文变量c，并且将输入序列的信息在该上下文变量中进行编码。

考虑由一个序列组成的样本（批量大小是1）。假设输入序列是$x_1,\dots ,x_T$，其中$x_t$是输入文本序列中的第t个词元。在时间步$t$，循环神经网络将词元$x_t$的输入特征向量$x_t$和$h_{t-1}$（即上一时间步的隐状态）转换为$h_t$（即
当前步的隐状态）。使用一个函数$f$来描述循环神经网络的循环层所做的变换：

$$ \mathbf{h}_t = f(\mathbf{x}_t,\mathbf{h}_t)$$

编码器通过选定的函数$q$，将所有时间步的隐状态转换为上下文变量：

$$ \mathbf{c} = q(\mathbf{h}_1, \dots , \mathbf{h}_T)$$

当$q(\mathbf{h}_1, \dots , \mathbf{h}_T)=\mathbf{h}_T$时，上下文变量仅仅是输入序列在最后时间步的隐状态$\mathbf{h}_T$。

