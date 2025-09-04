---
title: 自然语言处理（6）Attention Model
tags: DL NLP Transformer
typora-root-url: ./..
---

学习注意力机制（Attention Mechanism）、自注意力机制（Self-Attention）、多头注意力机制（Multi-Head Attention）。

<!--more-->

论文：2014，Neural Machine Translation by Jointly Learning to Align and Translate，带有注意力的编码解码

##### 1.论文解读

在对于传统的seq2seq模型，有一个潜在的问题就是编码器总是将所有出入信息编码为一个定长的隐向量，这可能会导致网络无法解决长句子，尤其是那些比训练语料还要长的句子，当输入的句子长度上升时，传统的编码器—解码器模型的表现急剧下降。

所以引入了权重，将输入句子编码为向量序列，在解码时根据权重自适应地选择这些向量的一个子集，这样就不用将源语句所有信息压缩到固定长度的向量。

解码器中隐状态计算方式为

$$ s_i = f(s_{i-1},y_{i-1},c_i)$$

计算方式根据选择的RNN模型进行设计，其中，上下文向量也有一个可训练权重。

在之前的seq2seq中，上下文向量是编码器最终的隐状态，而这篇论文将上下文向量$c_i$计算为标注$h_j$的加权和

$$ c_i = \sum_{j=1}^{T_x} \alpha_{ij} \cdot h_j $$

每个标注$h_j$的权重$\alpha_j$由下式计算

$$ \alpha_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})} = softmax(e_{ij}) $$

$$ e_{ij} = a(s_{i-1},h_j)=v_a^T tanh(W_a s_{i-1} + U_a h_j) $$

其中，$s_i$是解码器时间步$i$的隐状态，$h_i$是编码器时间步$i$的隐状态，$e_{ij}$是对齐模型，其他都是权重。

可以将取加权和的方法理解为计算期望标注，即对所有标注计算期望，第$i$个上下文向量$c_i$是标注在概率$\alpha_{ij}$下的期望标注。其中，$\alpha_{ij}$的作用相当于目标词$y_i$与源词$x_j$对齐或从源词$x_j$翻译而来的概率。

概率$\alpha_{ij}$或与其相关的分数$e_{ij}$反映了在决定下一个状态$s_i$和生成$y_i$时，标注$h_j$相对于之前的隐藏状态$s_{i - 1}$的重要性。直观地说，这实现了解码器中的注意力机制。解码器决定源语句的哪些部分需要关注。通过让解码器具备注意力机制，我们减轻了编码器将源语句的所有信息编码到固定长度向量中的负担。借助这种新方法，所有信息可以分布在整个标注序列中，解码器可以据此有选择地提取这些信息。

##### 2.注意力机制（Attention Mechanism）

第一节的论文模型中，通过为输入序列的不同部分分配不同的“权重”，使模型能够在生成输出的每一步都动态地、有选择性地关注输入中最相关的部分，这就是注意力机制，由于用到了加法计算，所以也被称为加性注意力（Additive Attention）。

现在对上述论文举个机器翻译的例子，再进行解释。

![](/images/NLP/17.png)

图中展示的是生成单词“machine”的计算方式，将前一个时间步的输出状态$q_2$和编码器的输出进行注意力计算，得到当前时刻的包含重要信息的上下文向量

$$ c_i = \sum_{i=1}^{4} \alpha_{i} \cdot h_i = softmax(\lbrack s(q_2,h_1),s(q_2,h_2),s(q_2,h_3),s(q_2,h_4) \rbrack) $$

其中，$s(q_i,h_j)$与论文中的$e_{ij} = a(s_{i-1},h_j)$都是对齐模型，这里把它称为打分函数。之后，将这个上下文向量、上个时刻的输出、上个时刻的隐状态$q_2$进行计算作为当前时刻的输出。

对于更一般的模型，可以用下图表示注意力机制的计算过程。

![](/images/NLP/18.png)

假设现在我们要对一组输入$H= \lbrack h_1,h_2,h_3, \cdots ,h_n \rbrack$使用注意力机制计算重要的内容（包含重要信息的上下文向量），这里会需要一个查询向量$q$，表示当前处理的信息是什么，在上述的机器翻译任务中，当前处理的信息是$q_2$。然后通过一个打分函数来计算查询向量$q$和每个输入$h_i$之间的相关性，得到一个分数。使用softmax对分数归一化，就得到了查询向量$q$在各个输入$h_i$上的注意力分布$a =  \lbrack a_1,a_2,a_3, \cdots ,a_n \rbrack$，其中每一项和原始输入$H  =\lbrack h_1,h_2,h_3, \cdots ,h_n \rbrack$一一对应。

$$ \alpha_{i} = softmax(s(h_i,q)) = \frac{exp(s(h_i,q))}{\sum_{k=1}^{T_x}exp(s(h_j,q))} $$

最后，根据这些注意力分布可以有选择性的从输入信息中提取信息，最经典的就是加性注意力，得到了模型当前应该关注的内容c。

$$c_i = \sum_{i=1}^{n} \alpha_{i} \cdot h_i$$

这里也记录一些其他的打分函数：

- 加性模型: $s(h,q) = v^T \tanh(Wh + Uq)$
- 点积模型: $s(h,q) = h^T q$
- 缩放点积模型: $s(h,q) = \frac{h^T q}{\sqrt{D}}$
- 双线性模型: $s(h,q) = h^T W q$

其中，W、U和v都是可学习的权重参数，D是输入向量的维度，进行缩放是为了防止点积结果过大导致 softmax 函数梯度消失。

##### 3.自注意力机制（self-Attention）

在上面的论文中，有三个部分需要考虑，第一个是当前需要处理的信息或者状态q，第二个是输入序列每个元素映射后的标签，用于和q相匹配，两者用来计算打分函数，第三个是输入序列的实际内容或特征表示，即真实的信息，上下文向量通过给真实信息分配权重，输出最重要的信息。这三个部分构成了注意力机制的核心。

对这三部分命名如下：

- 查询（Query）：当前模型正在处理的状态或信息。
- 键（Key）：输入序列每个元素映射后的标签。
- 值（Value）：输入序列每个元素的实际内容或特征表示。

在Neural Machine Translation by Jointly Learning to Align and Translate论文中，Q对应解码器上一时刻的状态，K和V对应输入序列每个元素映射后的标签。

而在自注意力机制中，Q、K、V都来自原始输入序列$X$，通过对原始序列线性变换得到，即

$$ Q = W_q \cdot X $$

$$ K = W_k \cdot X $$

$$ V = W_v \cdot X $$

三个线性变换，只有权重矩阵不一样。

之后对Q合K使用打分函数+sofrmax计算得分，将得分权重与V计算得到当前位置的Attention向量、重要信息c

$$ Attention(Q,K,V)=c = softmax(s(Q,K)) \cdot V $$

该公式就是自注意力机制的核心部分。

##### 4.多头注意力机制（Multi-Head Attention）

回顾刚刚的单个自注意力机制，会发现没有什么可以学习的参数，就只是三个权重和原始序列的线性内积，注意力只能在同一个子空间进行计算，这意味着模型只能从 “一个角度” 理解输入序列中元素的关联，无法并行捕捉不同维度、不同类型的依赖关系。

为了让模型在多个独立的 “表示子空间” 中并行计算注意力，从而从不同维度、不同视角挖掘输入序列中的关联信息，避免单一子空间的信息捕捉盲区。

多头注意力机制通过四个步骤，实现了这个作用

（1）将Q、K、V按头的数量h拆分为h组独立的子向量，每组维度为$d_k = d_{model} / h$，即$Q =  \lbrack Q_1,Q_2, \cdots ,Q_h \rbrack$、$K =  \lbrack K_1,K_2, \cdots ,K_h \rbrack$、$V =  \lbrack V_1,V_2, \cdots ,V_h \rbrack$。

（2）对每组子向量$(Q_i, K_i, V_i)$，独立执行标准的自注意力计算（相似度得分→softmax 权重→加权求和），得到h组局部输出$Z_1, Z_2, ..., Z_h$（每组维度为$n \times d_k$，n为序列长度）。

（3）将h组局部输出按列拼接，得到维度为$n \times (h \cdot d_k) = n \times d_{model}$的全局向量（恢复为原始输入维度，保证后续模块可兼容）。

（4）将拼接后的向量通过一个最终的线性变换矩阵$W^O$，得到多头注意力的最终输出Z（整合多头子空间的信息，形成统一的特征表示）。

也就是说，多头注意力机制和为了自注意力机制而设计的，这也是Transformer的核心内容。

![](/images/NLP/20.png)

##### 5.加性注意力实现

沿用机器翻译的例子，增加Neural Machine Translation by Jointly Learning to Align and Translate中的注意力机制。

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

# 在序列中屏蔽不相关的项
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

# 通过在最后⼀个轴上掩蔽元素来执⾏softmax操作
def masked_softmax(X, valid_lens):
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
        
# 加性注意⼒
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使⽤⼴播⽅式进⾏求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)
        
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

# 带有注意⼒机制解码器的基本接⼝
class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
    
    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)
    
    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(
            query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
       
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
    # 对输入句子小写，分词，映射词表，加<eos>
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
lr, num_epochs = 0.005, 250
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps, zip_path, file_in_zip)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
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
    translation, dec_attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {bleu(translation, fra, k=2):.3f}')
~~~

输出结果与之前的机器翻译别无二致。此外，也可以使用热力图查看Q和K的权重关系。

~~~
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            # 将 PyTorch 张量转换为 NumPy 数组
            matrix_np = matrix.detach().numpy()
            # 绘制热力图
            pcm = ax.imshow(matrix_np, cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    # 添加颜色条
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()

attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq],
                              0).reshape((1, 1, -1, num_steps))

show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
              xlabel='Key positions', ylabel='Query positions')
~~~

![](/images/NLP/19.png)

