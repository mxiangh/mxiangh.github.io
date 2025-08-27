---
title: 自然语言处理（5）机器翻译案例：英—法
tags: DL RNN NLP
typora-root-url: ./..
---

实现自然语言处理中的机器翻译案例。

<!--more-->

##### 1.机器翻译与seq2seq

机器翻译（machine translation）指的是将序列从⼀种语言自动翻译成另⼀种语言。机器翻译有两大流派，一个是统计机器翻译（statistical machine translation），另一个是神经机器翻译（neural machine translation），本篇及之后都以神经机器翻译为主。

机器翻译中的输入序列和输出序列都是长度可变的。为了解决这类问题，本篇使用seq2seq模型，在最开始的seq2seq模型中，使用的是两个LSTM，本篇使用两个GRU构造一个“编码器—解码器”架构。

##### 2.数据预处理

###### 2.1 数据集下载

Tatoeba项目的双语句子对组成的“英一法”数据集，下载网址：http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip

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
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8, zip_path=zip_path, file_in_zip=file_in_zip)

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

为了连续生成输出序列的词元，独立的循环神经网络解码器是基于输入序列的编码信息和输出序列已经看见的或者生成的词元来预测下一个词元。解码器当前时间步的输出依赖前一时间步的输出和编码器输出的上下文向量。

![](/images/NLP/12.png)

图中，\<eos\>表示序列结束词元，一旦输出序列生成此词元，模型就会停止预测。\<bos\>表示序列开始词元，是解码器的输入序列的第一个词元。其次，使用循环神经网络编码器最终的隐状态来初始化解码器的隐状态。此外，也可以让编码器最终的隐状态在每一个时间步都作为解码器的输入序列的一部分。

注：编码器最终的隐状态可以看作是输入序列的编码信息，用这个编码信息初始解码器隐状态，等于将输入序列信息的上下文向量传递给解码器。

###### 3.1 编码器

编码器将长度可变的输入序列转换成形状固定的上下文变量c，并且将输入序列的信息在该上下文变量中进行编码。

考虑由一个序列组成的样本（批量大小是1）。假设输入序列是$x_1,\dots ,x_T$，其中$x_t$是输入文本序列中的第t个词元。在时间步$t$，循环神经网络将词元$x_t$的输入特征向量$x_t$和$h_{t-1}$（即上一时间步的隐状态）转换为$h_t$（即
当前步的隐状态）。使用一个函数$f$来描述循环神经网络的循环层所做的变换：

$$ \mathbf{h}_t = f(\mathbf{x}_t,\mathbf{h}_{t-1})$$

编码器通过选定的函数$q$，将所有时间步的隐状态转换为上下文变量：

$$ \mathbf{c} = q(\mathbf{h}_1, \dots , \mathbf{h}_T)$$

当$q(\mathbf{h}_1, \dots , \mathbf{h}_T)=\mathbf{h}_T$时，上下文变量仅仅是输入序列在最后时间步的隐状态$\mathbf{h}_T$。

使用多层门控循环单元（GRU）构建序列到序列的编码器，同时，使用嵌入层（embeddinglayer）来获得输入序列中每个词元的特征向量。嵌入层是自然语言处理中的概念，其行数等于输入词表的大小（vocab_size），其列数等于特征向量的维度（embed_size）。嵌入层有自己的权重，是一个可以训练的量，用于特定的表示每个输入，以便转为上下文向量。

在编码器接口中，我们只指定长度可变的序列作为编码器的输入X。任何继承这个Encoder基类的模型将完成代码实现。编码器接口的目的是规范化使用，本身没有任何计算作用。

~~~
from torch import nn

# 编码器-解码器架构的基本编码器接口
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        
    def forward(self, X, *args):
        raise NotImplementedError

# 用于序列到序列学习的循环神经网络编码器
class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌⼊层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)
        
    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第⼀个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state
~~~

测试一下。我们使用一个两层门控循环单元编码器，其隐藏单元数为16。给定一小批量的输入序列X（批量大小为4，时间步为7）。在完成所有时间步后，最后一层的隐状态的输出是一个张量（output由编码器的循环层返回)，其形状为（时间步数，批量大小，隐藏单元数）。

~~~
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
output.shape, state.shape
~~~

由于这里使用的是门控循环单元，所以在最后一个时间步的多层隐状态的形状是（隐藏层的数量，批量大小，隐藏单元的数量）。如果使用长短期记忆网络，state中还将包含记忆单元信息。

###### 3.2 解码器

编码器输出的上下文变量$c$对整个输入序列$x$进行编码，对于解码器每个时间步，当前时间步输出$y_t$的概率取决于先前的输出子序列$y_1,y_2,\dots ,y_{t-1}$和上下文变量$c$，即

$$ P(y_{t^{'}} \mid y_1,\dots ,y_{t^{'}-1},\mathbf{c}  )$$

我们用另一个循环神经网络模型化这种条件概率，在输出序列的任意时间步，循环神经网络将来自上一时间步的输出$y_{t-1}$和上下文向量$c$作为输入，然后在当前时间步将它们和上一隐状态$s_{t-1}$转换为隐状态$s_t$，用函数$g$表示解码层隐藏层的变换

$$ \mathbf{s}_{t^{'}} = g(y_{t^{'}-1},\mathbf{c}, \mathbf{s}_{t^{'}-1})$$

在获得解码器的隐状态之后，我们可以使用输出层和softmax操作来计算在时间步$t$时输出$y$的条件概率分布。

综上所述：

（1）当实现解码器时，直接使用编码器最后一个时间步的隐状态来初始化解码器的隐状态。这就要求使用循环神经网络实现的编码器和解码器具有相同数量的层和隐藏单元。

（2）为了进一步包含经过编码的输入序列的信息，上下文变量在所有的时间步与解码器的输入进行拼接（concatenate）。

（3）为了预测输出词元的概率分布，在循环神经网络解码器的最后一层使用全连接层来变换隐状态。

在下面的解码器接口中，新增一个init_state函数，用于将编码器的输出（enc_outputs）转换为编码后的状态。注意，此步骤可能需要额外的输入，例如：输入序列的有效长度。
为了逐个地生成长度可变的词元序列，解码器在每个时间步都会将输入（例如：在前一时间步生成的词元）和编码后的状态映射成当前时间步的输出词元。

~~~
# 编码器-解码器架构的基本解码器接⼝
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
        
    def forward(self, X, state):
        raise NotImplementedError
        
# 用于序列到序列学习的循环神经网络解码器
class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
        
    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # ⼴播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state
~~~

测试一下。解码器的输出形状变为（批量大小，时间步数，词表大小），其中张量的最后一个维度存储预测的词元分布。

~~~
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, state.shape
~~~

###### 3.3 “编码器-解码器”架构

总而言之，“编码器-解码器”架构包含了一个编码器和一个解码器，并且还拥有可选的额外的参数。在前向传播中，编码器的输出用于生成编码状态，这个状态又被解码器作为其输入的一部分。

~~~
# 编码器-解码器架构的基类
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
~~~

![](/images/NLP/13.png)

##### 4.损失函数

在每个时间步，解码器预测了输出词元的概率分布。类似于语言模型，可以使用softmax来获得分布，并通过计算交叉熵损失函数来进行优化。

将特定的填充词元被添加到序列的末尾，因此不同长度的序列可以以相同形状的小批量加载。但是，我们应该将填充词元的预测排除在损失函数的计算之外。

为此，我们可以使用下面的sequence_mask函数通过零值化屏蔽不相关的项，以便后面任何不相关预测的计算都是与零的乘积，结果都等于零。例如，如果两个序列的有效长度（不包括填充词元）分别为1和2，则第一个序列的第一项和第二个序列的前两项之后的剩余项将被清除为零。

~~~
# 在序列中屏蔽不相关的项
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
    
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, torch.tensor([1, 2]))
~~~

![](/images/NLP/14.png)

现在，我们可以通过扩展softmax交叉熵损失函数来遮蔽不相关的预测。最初，所有预测词元的掩码都设置为1。一旦给定了有效长度，与填充词元对应的掩码将被设置为0。最后，将所有词元的损失乘以掩码，以过滤掉损失中填充词元产生的不相关预测。

~~~
# 带遮蔽的softmax交叉熵损失函数
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
~~~

我们可以创建三个相同的序列来进行代码健全性检查，然后分别指定这些序列的有效长度为4、2和0。结果就是，第一个序列的损失应为第二个序列的两倍，而第三个序列的损失应为零。

~~~
loss = MaskedSoftmaxCELoss()
loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
~~~

![](/images/NLP/15.png)

##### 5.训练

使用之前训练语言模型时定义过的梯度裁剪，之后初始化，前向传播，反向传播，记录损失函数，都是常规操作。

~~~
# 梯度裁剪
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
            
# 训练序列到序列模型
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
        
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    all_loss = []
    for epoch in range(num_epochs):
        total_loss = 0.0  # 批量累计总损失
        total_tokens = 0  # 批量累计总词元数
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1) # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward() # 损失函数的标量进⾏“反向传播”
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                # 批量累计损失和词元数
                total_loss += l.sum()
                total_tokens += num_tokens
        all_loss.append((total_loss / total_tokens).cpu().numpy())
            
    print(f'loss {total_loss / total_tokens:.3f}')

    return all_loss
~~~

##### 6.预测

为了采用一个接着一个词元的方式预测输出序列，每个解码器当前时间步的输入都将来自于前一时间步的预测词元。与训练类似，序列开始词元（“\<bos\>”）在初始时间步被输入到解码器中。当输出序列的预测遇到序列结束词元（“\<eos\>”）时，预测就结束了。

![12](/images/NLP/12.png)

~~~
# 序列到序列模型的预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使⽤具有预测最高可能性的词元，作为解码器在下⼀时间步的输⼊
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
~~~

##### 7.评估

通常使用BLEU(bilingual evaluation understudy）测量输出序列的质量，对于预测序列中的任意n元语法（n-grams），BLEU的评估都是这个n元语法是否出现在标签序列中。

$$ BLEU = exp(min(0,1-\frac{len_{label}}{len_{pred}})) \prod_{n=1}^k p_n^{\frac{1}{2^n}}$$

注：n元语法是指 “连续 n 个词（或字符）的组合”。对于a、b、c、d，一元语法是a、b、c、d；二元语法是(a,b)、(b,c)、(c,d)，以此类推，n最多为词元数量。

其中，label表示标签序列，pred表示预测序列，k是用于匹配的最长的n元语法。$p_n$表示n元语法的精确度，是 “预测序列与标签序列中匹配的 n 元语法数量” 除以 “预测序列中 n 元语法的总数量” 的比值。例如，给定标签序列A、B、C、D、E、F和预测序列A、B、B、C、D，我们有$p_1$=4/5、$p_2$ =3/4、$p_3$ =1/3和$p_4$=0。

~~~
import math

# 评估模型BLEU
def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
            label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
~~~

##### 8.代码汇总

~~~
# 代码汇总
import zipfile
import collections
import torch
import torch.utils.data as data
import math
from torch import nn
import matplotlib.pyplot as plt

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

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
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

# 截断或填充文本序列
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
    
# 编码器-解码器架构的基本编码器接口
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        
    def forward(self, X, *args):
        raise NotImplementedError

# 用于序列到序列学习的循环神经网络编码器
class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌⼊层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)
        
    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第⼀个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state
        
# 编码器-解码器架构的基本解码器接口
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
        
    def forward(self, X, state):
        raise NotImplementedError
        
# 用于序列到序列学习的循环神经网络解码器
class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
        
    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # ⼴播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state
        
# 编码器-解码器架构的基类
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
        
# 在序列中屏蔽不相关的项
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

# 带遮蔽的softmax交叉熵损失函数
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
        
# 梯度裁剪
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
            
# 训练序列到序列模型
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
        
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    all_loss = []
    for epoch in range(num_epochs):
        total_loss = 0.0  # 批量累计损失
        total_tokens = 0  # 批量累计词元数
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1) # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward() # 损失函数的标量进⾏“反向传播”
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                # 批量累计损失和词元数
                total_loss += l.sum()
                total_tokens += num_tokens
        all_loss.append((total_loss / total_tokens).cpu().numpy())
            
    print(f'loss {total_loss / total_tokens:.3f}')

    return all_loss
    
# 序列到序列模型的预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使⽤具有预测最高可能性的词元，作为解码器在下⼀时间步的输⼊
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
   
# 评估模型BLEU
def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
            label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
    
# 压缩包路径（根据实际文件位置修改）
zip_path = "fra-eng.zip"
# 压缩包内目标文件的路径（注意目录结构是否正确）
file_in_zip = "fra-eng/fra.txt"

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs = 0.005, 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps, zip_path, file_in_zip)

encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,dropout)
net = EncoderDecoder(encoder, decoder)
all_loss = train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# 绘制损失随迭代次数下降的折线图
plt.plot(range(1, num_epochs + 1), all_loss, label='train')  # 横轴：epoch，纵轴：loss
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.show()   
    
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, _ = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
~~~

![](/images/NLP/16.png)

