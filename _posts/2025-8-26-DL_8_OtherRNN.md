---
title: 深度学习（8）深度RNN和双向RNN
tags: DL RNN
typora-root-url: ./..
---

更深的循环神经网络和双向的循环神经网络。

<!--more-->

 ##### 1.深度循环神经网络

之前学习的RNN、LSTM、GRU及实现的语言模型，都是单个隐藏状态层，本篇介绍多层循环神经网络。

![](/images/RNN/6.png)

上图是一个具有$L$个隐藏层的深度循环神经网络，每个隐状态都连续地传递到当前层的下一个时间步和下一层的当前时间步。

假设在时间步$t$有一个小批量的输入数据$X$（样本数：n，每个样本中的输入数：d）。同时，将当前时间步的$l$隐藏层的隐状态设为$H_t^{(l)} $（隐藏单元数：h），输出层变量设为$O$（输出数：q）。设置$H_t^{(0)}=X_t$，第$l$个隐藏层的隐状态使用激活函数$g$，则：
$$\mathbf{H}_t^{(l)}=g(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)} + \mathbf{b}_h^{(l)})$$
其中，权重和参数是第$l$个隐藏层的模型参数。最后，输出层的计算仅基于第$l$个隐藏层最终的隐状态（最后一个隐藏层的最后一个隐状态）：

$$ \mathbf{O}_t = \mathbf{H}_{t}^{(L)} \mathbf{W}_{hq} + \mathbf{b}_q$$

与多层感知机一样，隐藏层数目$L$和隐藏单元数目$h$都是超参数。

##### 2.简洁实现多层LSTM

这里沿用简洁LSTM的大部分代码，只修改两行代码以增加隐藏层数num_layers。

对于GRU可以类似地添加隐藏层数。

~~~
# LSTM添加隐藏层数
num_layers = 2

lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
~~~

多层LSTM完整代码如下。

~~~
import requests
import re
import collections
import random
import torch
import math
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# # 获取数据
# url = "https://www.gutenberg.org/cache/epub/35/pg35.txt" 
# response = requests.get(url)
# with open('TimeMachine.txt', 'w', encoding='utf-8') as f:
#     f.write(response.text)

# 数据清洗
def read_time_machine():
    with open('TimeMachine.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 对每行进行清洗
    # 1.re.sub：去除非字母字符
    # 2.strip：去除字符串首尾的空白字符
    # 3.lower：所有大写字母转为小写字母
    cleaned_lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    # 对处理后的列表，删除里面的空字符串
    return [line for line in cleaned_lines if line]

# 词元化
def tokenize(lines, token='word'):
    # 将⽂本⾏拆分为单词或字符词元
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)
        
# 构建词表
class Vocab:
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
            
# 统计词元的频率
def count_corpus(tokens):
    # 这⾥的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成⼀个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 返回时光机器数据集的词元索引列表和词表（词元为单词），max_tokens用于截断
def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines,'char')
    vocab = Vocab(tokens)
    
    # 因为时光机器数据集中的每个⽂本⾏不⼀定是⼀个句⼦或⼀个段落，
    # 所以将所有⽂本⾏展平到⼀个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# 使⽤随机抽样⽣成⼀个⼩批量⼦序列
# corpus是序列列表，batch_size是批量大小，num_steps是时间步数
def seq_data_iter_random(corpus, batch_size, num_steps):
    # 从随机偏移量开始对序列进⾏分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # ⻓度为num_steps的⼦序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来⾃两个相邻的、随机的、⼩批量中的⼦序列不⼀定在原始序列上相邻
    random.shuffle(initial_indices)
    
    def data(pos):
        # 返回从pos位置开始的⻓度为num_steps的序列
        return corpus[pos: pos + num_steps]
        
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这⾥，initial_indices包含⼦序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
        
# 使⽤顺序分区⽣成⼀个⼩批量⼦序列
# corpus是序列列表，batch_size是批量大小，num_steps是时间步数
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
        
# 加载序列数据的迭代器
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
        
# 返回数据迭代器和词表
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# 循环神经⽹络模型
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大⼩,隐藏单元数)
        # 它的输出形状是(时间步数*批量大⼩,词表大⼩)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

# 预测新字符
# prefix：输入字符；num_preds：预测字符数；net：使用的网络模型；vocab：词表
def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]: # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds): # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

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

# 小批量随机梯度下降
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练一个迭代周期
def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state = None
    total_loss = 0.0  # 累计总损失
    total_tokens = 0  # 累计总词元数
    
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第⼀次迭代或使⽤随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调⽤了mean函数
            updater(batch_size=1)
            
        # 累计损失和词元数
        total_loss += l.item() * y.numel()
        total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)
    
# 训练模型
def train(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    perplexities = []  # 存储每次迭代的困惑度
    
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    
    # 训练和预测
    for epoch in range(num_epochs):
        ppl = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        perplexities.append(ppl)  # 记录当前epoch的困惑度
        
        # 每50个周期打印一次中间结果
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Perplexity: {ppl:.5f}")

    # 训练结束：输出最终结果
    print(f"Final Perplexity: {ppl:.5f} (on {device})")
    return perplexities

# 加载数据集
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# 词表大小，隐藏节点数，隐藏层数
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
# 指定GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 迭代次数，学习率
num_epochs, lr = 500, 1
num_inputs = vocab_size
# 将之前的RNN隐藏层替换为LSTM隐藏层
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)

perplexities = train(model, train_iter, vocab, lr, num_epochs, device)

# 绘制困惑度随迭代次数下降的折线图
plt.plot(range(1, num_epochs + 1), perplexities, label='train')  # 横轴：epoch，纵轴：困惑度
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.show()

# 用于预测下一个词元的序列
prefix = "time traveller"
# 预测50个字符，训练参数存放在model中
print(predict(prefix, 50, model, vocab, device))
~~~

##### 3.双向循环神经网络（Bidirectional RNN，BRNN）

在序列学习中，我们以往假设的目标是：在给定观测的情况下（例如，在时间序列的上下文中或在语言模型的上下文中），对下一个输出进行建模。虽然这是一个典型情景，但不是唯一的。

考虑以下三个在文本序列中填空的任务：

- 我_ _ _。
- 我_ _ _饿了。
- 我_ _ _饿了，我可以吃半头猪。

很明显，每个短语的“下文”传达了重要信息（如果有的话），而这些信息关乎到选择哪个词来填空，所以无法利用这一点的序列模型将在相关任务上表现不佳。例如，如果要做好命名实体识别（例如，识别“Green”指的是“格林先生”还是绿色），不同长度的上下文范围重要性是相同的。

传统的 RNN 在处理序列数据时，信息是单向流动的，从序列的起始位置向结束位置传递，只能利用过去的信息 。而双向循环神经网络允许信息在序列中双向流动，即同时从序列的开头向结尾（正向传播）和从结尾向开头（反向传播）进行处理，从而使模型在每个时刻都能同时利用到过去和未来的信息。

![](/images/RNN/7.png)

对于任意时间步$t$，给定一个小批量的输入数据$X$样本数：n，每个示例中的输入数：d），并且令隐藏层激活函数为$g$。在双向架构中，我们设该时间步的前向和反向隐状态分别为$\overrightarrow{H}$和$\overleftarrow{H} $，其中$h$是隐藏单元的数目。前向和反向隐状态的更新如下：

$$ \overrightarrow{\mathbf{H}}_t = g(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)} + \mathbf{b}_h^{(f)}) $$

$$ \overleftarrow{\mathbf{H}}_t = g(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)} + \mathbf{b}_h^{(b)}) $$

$f$代表前，$b$代表反。

接着，将前向和反向隐状态拼在一起连接起来，获得当前时间步需要送入输出层的隐状态$H_t$。在具有多个隐藏层的深度双向循环神经网络中，该信息作为输入传递到下一个双向层。

最后，输出层计算得到输出：

$$ \mathbf{O}_t = \mathbf{H}_{t} \mathbf{W}_{hq} + \mathbf{b}_q $$

双向循环神经网络的一个关键特性是：使用来自序列两端的信息来估计输出。也就是说，我们使用来自过去和未来的观测信息来预测当前的观测。

但是在对下一个词元进行预测的情况中，这样的模型并不是我们所需的。因为在预测下一个词元时，我们终究无法知道下一个词元的下文是什么，所以将不会得到很好的精度。具体地说，在训练期间，我们能够利用过去和未来的数据来估计现在空缺的词；而在测试期间，我们只有过去的数据，因此精度将会很差。

另一个严重问题是，双向循环神经网络的计算速度非常慢。其主要原因是网络的前向传播需要在双向层中进行前向和后向递归，并且网络的反向传播还依赖于前向传播的结果。因此，梯度求解将有一个非常长的链。

基于这两个问题的存在，双向层的使用在实践中非常少。

可以在LSTM层添加参数bidirectional=True，运用双向LSTM，但是这并不是一个正确的选择，如果将其用于语言模型预测将会得到一个糟糕的结果。

~~~
# 通过设置“bidirective=True”来定义双向LSTM模型
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
~~~

