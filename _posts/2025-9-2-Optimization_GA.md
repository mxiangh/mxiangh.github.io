---
title: 优化算法（2）遗传算法GA
tags: Optimization Algorithm
typora-root-url: ./..
---

遗传算法（Genetic Algorithm，GA）是一种启发式搜索算法，它模仿自然选择的过程来解决优化和搜索问题。

<!--more-->

##### 1.遗传算法对应概念

遗传算法试图找到给定问题的最佳解。达尔文进化论保留了种群的个体性状，而遗传算法则保留了针对给定问题的候选解集合（也称为individuals）。这些候选解经过迭代评估 (evaluate)，用于创建下一代解。更优的解有更大的机会被选择，并将其特征传递给下一代候选解集合。这样，随着世代更新，候选解集合可以更好地解决当前的问题。

- 基因型 (Genotype)：在自然界中，通过基因型表征繁殖，繁殖和突变，基因型是组成染色体的一组基因的集合。在遗传算法中，每个个体都由代表基因集合的染色体构成。例如，一条染色体可以表示为二进制串，其中每个位代表一个基因。
- 种群 (Population)：遗传算法保持大量的个体 (individuals)——针对当前问题的候选解集合。由于每个个体都由染色体表示，因此这些种族的个体可以看作是染色体集合。
- 适应度函数 (Fitness function)：在算法的每次迭代中，使用适应度函数(也称为目标函数)对个体进行评估。目标函数是用于优化的函数或试图解决的问题。适应度得分更高的个体代表了更好的解，其更有可能被选择繁殖并且其性状会在下一代中得到表现。随着遗传算法的进行，解的质量会提高，适应度会增加，一旦找到具有令人满意的适应度值的解，终止遗传算法。
- 选择 (Selection)：在计算出种群中每个个体的适应度后，使用选择过程来确定种群中的哪个个体将用于繁殖并产生下一代，具有较高值的个体更有可能被选中，并将其遗传物质传递给下一代。仍然有机会选择低适应度值的个体，但概率较低。这样，就不会完全摒弃其遗传物质。
- 交叉 (Crossover)：为了创建一对新个体，通常将从当前代中选择的双亲样本的部分染色体互换(交叉)，以创建代表后代的两个新染色体。此操作称为交叉或重组。
- 突变 (Mutation)：突变操作的目的是定期随机更新种群，将新模式引入染色体，并鼓励在解空间的未知区域中进行搜索。突变可能表现为基因的随机变化。变异是通过随机改变一个或多个染色体值来实现的。例如，翻转二进制串中的一位。

##### 2.算法详解

###### 2.1 超参数

（1）种群大小：候选解的个数，随机生成。

（2）编码长度：如果使用二进制编码，需要设置编码长度。

（3）交叉概率：使用交叉方法生成新的候选解的概率，可以不设置，默认必使用。

（4）变异概率：使用变异方法生成新的候选解的概率，可以不设置，默认必使用。

（5）迭代次数：算法迭代的最大次数。

###### 2.2 产生新解

（1）适应度函数：计算当前候选解每个解的值，遗传算法一般求解最大值，所以还需要对值进行处理（最小值取倒数）。

（2）选择：

a.轮盘赌选择：将适应度函数归一化转为概率，按照适应度函数的概率选择个体，概率越大，被选择的概率越大。

b.锦标赛选择：随机选择k个个体，从k个个体选择适应度最高的个体进入下一代。

c.精英选择：选择种群中适应度最高的k个个体直接进行下一代，其余个体通过其他选择方法选择。

（3）交叉：随机选择父代，按照初始设定的概率，选择是否生成子代。

（4）变异：随机选择个体，按照按照初始设定的概率，选择是否异变。

通常在一次迭代中，适应度函数计算、选择、交叉、变异会按顺序依次进行，但是这不是必然的，可以只交叉不变异，也可以只变异不交叉，还可以前50次迭代交叉、后50次迭代变异，又或者在交叉的前提下再选择是否变异。

###### 2.3 算法流程

![](/images/Optimization/4.png)

##### 3.案例

###### 3.1 函数求最优

f (x) = x・sin (10π・x) + 2.0在区间 [-1, 3] 上的最小值

~~~
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["font.family"] = ["SimHei"]  # 支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# -------------------------- 1. 初始化参数与目标函数 --------------------------
# 函数定义
def target_function(x):
    return x * np.sin(10 * np.pi * x) + 2.0


# 算法参数
MIN_X = -1.0  # x最小值
MAX_X = 3.0  # x最大值
POP_SIZE = 50  # 种群大小
CHROM_LEN = 20  # 二进制编码长度
CROSS_PROB = 0.8  # 交叉概率提高到0.8
MUTATE_PROB = 0.4  # 变异概率调整为0.4
MAX_ITER = 100  # 最大迭代次数
PRECISION = (MAX_X - MIN_X) / (2 ** CHROM_LEN - 1)  # 计算精度

# 预生成目标函数曲线数据（用于可视化）
x_full = np.linspace(MIN_X, MAX_X, 1000)
y_full = target_function(x_full)


# -------------------------- 2. 遗传算法核心函数 --------------------------
# 解码
def binary_to_decimal(binary):
    """二进制染色体解码为十进制x值（映射到[-1, 3]）"""
    decimal = 0
    for bit in binary:
        decimal = decimal * 2 + bit
    # 将十进制数映射到目标区间
    x = MIN_X + decimal * (MAX_X - MIN_X) / (2 ** CHROM_LEN - 1)
    return x


# 编码
def init_population():
    """初始化种群：生成POP_SIZE个CHROM_LEN位的二进制个体"""
    population = []
    for _ in range(POP_SIZE):
        # 随机生成0/1二进制数组
        chromosome = np.random.randint(0, 2, size=CHROM_LEN)
        population.append(chromosome)
    return population


# 计算适应度
def calculate_fitness(population):
    """计算种群中每个个体的适应度（目标函数值）"""
    fitness = []
    for chromosome in population:
        x = binary_to_decimal(chromosome)
        y = target_function(x)
        fitness.append(y)
    return np.array(fitness)


# 选择
def selection(population, fitness):
    """轮盘赌选择：适应度越小（函数值越小），被选中概率越高"""
    # 处理负适应度值的问题：将所有适应度值平移到非负区间
    min_fitness = np.min(fitness)
    if min_fitness < 0:
        adjusted_fitness = fitness - min_fitness + 1e-6  # +1e-6防止为0
    else:
        adjusted_fitness = fitness + 1e-6  # 确保没有0值

    # 适应度取倒数（将最小化问题转为最大化问题）
    inv_fitness = 1.0 / adjusted_fitness
    selection_prob = inv_fitness / np.sum(inv_fitness)  # 选择概率

    # 按概率选择个体（可重复选择，模拟“优胜劣汰”）
    selected_indices = np.random.choice(len(population), size=POP_SIZE, p=selection_prob)
    new_population = [population[i] for i in selected_indices]
    return new_population


# 交叉并变异
def crossover_and_maybe_mutate(population):
    """先进行交叉（概率0.8），若交叉则可能进行变异（概率0.4）"""
    new_population = []
    for i in range(0, POP_SIZE, 2):  # 两两配对
        parent1 = population[i]
        parent2 = population[i + 1] if (i + 1) < POP_SIZE else parent1  # 处理奇数种群

        if np.random.random() < CROSS_PROB:
            # 执行交叉
            cross_point = np.random.randint(1, CHROM_LEN - 1)  # 随机选择交叉点
            child1 = np.hstack([parent1[:cross_point], parent2[cross_point:]])
            child2 = np.hstack([parent2[:cross_point], parent1[cross_point:]])

            # 仅在交叉发生的前提下才进行变异（概率0.4）
            if np.random.random() < MUTATE_PROB:
                # 对child1进行变异
                for j in range(CHROM_LEN):
                    if np.random.random() < MUTATE_PROB:
                        child1[j] = 1 - child1[j]
                # 对child2进行变异
                for j in range(CHROM_LEN):
                    if np.random.random() < MUTATE_PROB:
                        child2[j] = 1 - child2[j]

            new_population.extend([child1, child2])
        else:
            # 不交叉，直接保留父母
            new_population.extend([parent1, parent2])

    return new_population[:POP_SIZE]  # 确保种群大小不变


# -------------------------- 3. 迭代过程与数据记录 --------------------------
# 初始化种群与最优值跟踪
population = init_population()
global_best_y = float('inf')  # 全局最优值（初始为无穷大）
global_best_x = None  # 全局最优x
iter_history = []  # 迭代次数记录
best_y_history = []  # 每代最优值记录（仅更新更优值）
best_x_history = []
all_populations = []  # 存储每代种群用于可视化

# 执行遗传算法迭代
for iter in range(MAX_ITER):
    # 保存当前种群
    all_populations.append(population.copy())

    # 1. 计算当前种群适应度
    fitness = calculate_fitness(population)
    current_min_idx = np.argmin(fitness)
    current_min_y = fitness[current_min_idx]
    current_min_x = binary_to_decimal(population[current_min_idx])

    # 2. 更新全局最优值（仅保留更优值）
    if current_min_y < global_best_y:
        global_best_y = current_min_y
        global_best_x = current_min_x

    # 3. 记录迭代数据
    iter_history.append(iter)
    best_y_history.append(global_best_y)  # 始终记录当前全局最优
    best_x_history.append(global_best_x)

    # 4. 执行遗传操作（选择→交叉+可能的变异）
    population = selection(population, fitness)
    population = crossover_and_maybe_mutate(population)  # 合并了交叉和可能的变异操作

# 提取每代种群的x/y值（用于可视化种群分布）
population_x_history = []
population_y_history = []
for pop in all_populations:
    x_list = [binary_to_decimal(chrom) for chrom in pop]
    y_list = [target_function(x) for x in x_list]
    population_x_history.append(x_list)
    population_y_history.append(y_list)

# -------------------------- 4. 动态可视化（双图联动） --------------------------
# 创建画布与子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 子图1：目标函数曲线 + 种群分布
ax1.plot(x_full, y_full, 'b-', linewidth=1.5, label='目标函数 f(x)')
scatter_pop = ax1.scatter([], [], c='red', s=30, alpha=0.6, label='当前种群')
scatter_best = ax1.scatter([], [], c='orange', s=100, marker='*', label='全局最优')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('遗传算法种群分布（迭代过程）', fontsize=14)
ax1.set_xlim(MIN_X - 0.1, MAX_X + 0.1)
ax1.set_ylim(np.min(y_full) - 0.5, np.max(y_full) + 0.5)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：迭代次数 vs 全局最优值
line_best = ax2.plot([], [], 'g-', linewidth=2, label='全局最优值')[0]
ax2.set_xlabel('迭代次数', fontsize=12)
ax2.set_ylabel('f(x) 最小值', fontsize=12)
ax2.set_title('迭代过程中全局最优值变化', fontsize=14)
ax2.set_xlim(0, MAX_ITER)
ax2.set_ylim(np.min(best_y_history) - 0.2, np.max(best_y_history) + 0.2)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 添加文本标注（显示当前迭代次数与最优值）
text_iter = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def update(frame):
    """动画更新函数：每帧更新两个子图的数据"""
    # 1. 更新子图1（种群分布）
    current_x = population_x_history[frame]
    current_y = population_y_history[frame]
    scatter_pop.set_offsets(np.c_[current_x, current_y])  # 更新种群散点
    scatter_best.set_offsets([[best_x_history[frame], best_y_history[frame]]])  # 更新全局最优

    # 2. 更新子图2（最优值曲线）
    line_best.set_data(iter_history[:frame + 1], best_y_history[:frame + 1])

    # 3. 更新文本标注
    text_iter.set_text(f'迭代次数: {frame + 1}\n当前最优值: {best_y_history[frame]:.6f}\n最优x: {global_best_x:.6f}')

    return scatter_pop, scatter_best, line_best, text_iter


# 创建动画（interval=500控制速度，单位ms）
ani = FuncAnimation(fig, update, frames=MAX_ITER, interval=500,
                    blit=True, repeat=False)

# 显示动画
plt.tight_layout()
plt.show()

# 输出最终结果
print(f"遗传算法迭代完成！")
print(f"搜索范围：x ∈ [{MIN_X}, {MAX_X}]")
print(f"迭代次数：{MAX_ITER}")
print(f"交叉概率：{CROSS_PROB}，变异概率：{MUTATE_PROB}（仅在交叉后应用）")
print(f"最终最优x值：{global_best_x:.6f}")
print(f"最终最优f(x)值：{global_best_y:.6f}")
~~~

###### 3.2 TSP旅行商问题

~~~
# 导入需要用到的包
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

plt.rcParams["font.family"] = ["SimHei"]  # 支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 初始化参数
species = 200
iters = 5000

def getListMaxNumIndex(num_list, topk=int(0.2 * species)):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    num_dict = {}
    for i in range(len(num_list)):
        num_dict[i] = num_list[i]
    res_list = sorted(num_dict.items(), key=lambda e: e[1])
    max_num_index = [one[0] for one in res_list[::-1][:topk]]
    return max_num_index


# 适应度函数      对路径的所有距离进行求和得到distance
def calfit(trip, num_city):
    total_dis = 0
    for i in range(num_city):
        cur_city = trip[i]
        next_city = trip[i + 1] % num_city
        temp_dis = distance[cur_city][next_city]
        total_dis = total_dis + temp_dis
    return 1 / total_dis


# 计算城市之间的距离和
def dis(trip, num_city):
    total_dis = 0
    for i in range(num_city):
        cur_city = trip[i]
        next_city = trip[i + 1] % num_city
        temp_dis = distance[cur_city][next_city]
        total_dis = total_dis + temp_dis
    return total_dis


# 交叉函数
def crossover(father, mother):
    num_city = len(father)
    index_random = [i for i in range(num_city)]
    pos = random.choice(index_random)
    son1 = father[0:pos]
    son2 = mother[0:pos]
    son1.extend(mother[pos:num_city])
    son2.extend(father[pos:num_city])
    ## 处理子代中的重复元素
    index_duplicate1 = []
    index_duplicate2 = []

    for i in range(pos, num_city):
        for j in range(pos):
            if son1[i] == son1[j]:
                index_duplicate1.append(j)
            if son2[i] == son2[j]:
                index_duplicate2.append(j)
    num_index = len(index_duplicate1)
    for i in range(num_index):
        son1[index_duplicate1[i]], son2[index_duplicate2[i]] = son2[index_duplicate2[i]], son1[index_duplicate1[i]]

    return son1, son2


# 变异函数
def mutate(sample):
    num_city = len(sample)
    part = np.random.choice(num_city, 2, replace=False)
    if part[0] > part[1]:
        max_ = part[0]
        min_ = part[1]
    else:
        max_ = part[1]
        min_ = part[0]
    after_mutate = sample[0:min_]
    temp_mutate = list(reversed(sample[min_:max_]))
    after_mutate.extend(temp_mutate)
    after_mutate.extend(sample[max_:num_city])
    return after_mutate


# Haversine距离计算函数（经纬度之间的距离，单位：千米）
def haversine(lon1, lat1, lon2, lat2):
    # 将十进制度数转换为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球平均半径，单位为千米
    return c * r


# 读取城市位置数据
import datetime

starttime = datetime.datetime.now()

# 读取新的数据格式，包含city、lon、lat列
df = pd.read_csv('city_coord.csv', encoding='utf-8')
# 提取经纬度数据并显示城市分布
plot = plt.plot(df['lon'], df['lat'], '*')

# 计算各城市邻接矩阵（使用Haversine距离）
n = len(df)
distance = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        # 使用Haversine公式计算距离
        distance[i][j] = haversine(
            df.iloc[i, 1],  # 第i个城市的经度
            df.iloc[i, 2],  # 第i个城市的纬度
            df.iloc[j, 1],  # 第j个城市的经度
            df.iloc[j, 2]  # 第j个城市的纬度
        )

# 初始化种群，生成可能的解的集合
x = []
counter = 0
while counter < species:
    # 生成一个随机的路径
    dna = np.random.permutation(range(n)).tolist()
    start = dna[0]
    dna.append(start)  # 回到起点，形成闭环
    if dna not in x:
        x.append(dna)
        counter = counter + 1

ctlist = []  # 用于记录迭代次数
dislist = []  # 用于记录每次迭代后得到的最短路径长度
ct = 0
while ct < iters:
    ct = ct + 1
    f = []
    # 计算种群中每个个体的适应度值（路径总距离的倒数）
    for i in range(species):
        f.append(calfit(x[i], n))

    # 计算选择概率
    sig = sum(f)
    p = (f / sig).tolist()

    # 从种群中选择适应度较高的个体
    test = getListMaxNumIndex(p)
    testnum = len(test)
    newx = []
    # 将适应度较高的个体加入下一代种群
    for i in range(testnum):
        newx.append(x[test[i]])

    index = [i for i in range(species)]
    # 根据选择概率进行随机选择，将选择的个体加入下一代种群
    news = random.choices(index, weights=p, k=int(0.8 * species))
    newsnum = len(news)
    for i in range(newsnum):
        newx.append(x[news[i]])

    m = int(species / 2)
    for i in range(0, m):
        j = i + m - 1
        # 对父代进行交叉和变异
        numx = len(newx[0])
        if random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) < 8:
            # 80%的概率执行交叉操作
            tplist1 = newx[i][0:numx - 1]
            tplist2 = newx[j][0:numx - 1]
            crosslist1, crosslist2 = crossover(tplist1, tplist2)
            if random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) < 4:
                # 40%的概率执行变异操作
                crosslist1 = mutate(crosslist1)
                crosslist2 = mutate(crosslist2)
            end1 = crosslist1[0]
            end2 = crosslist2[0]
            crosslist1.append(end1)
            crosslist2.append(end2)
            newx[i] = crosslist1
            newx[j] = crosslist2
    x = newx

    # 计算新一代种群的适应度值和最短路径长度
    res = []
    res1 = []
    for i in range(species):
        res.append(calfit(x[i], n))
        res1.append(dis(x[i], n))
    # 计算最短路径长度的倒数，即当前最优解的适应度值
    result = 1 / max(res)
    result1 = min(res1)
    # 打印当前迭代次数、最优解的适应度值和最短路径长度
    print(ct)
    print(result)
    print(result1)
    ctlist.append(ct)
    dislist.append(result)

endtime = datetime.datetime.now()
print(f"运行时间: {endtime - starttime}")

# 绘制最优路径图
plk1 = []
plk2 = []
for i in range(len(x[0])):
    plk2.append(df.iloc[x[0][i], 2])  # 纬度
    plk1.append(df.iloc[x[0][i], 1])  # 经度
plot = plt.plot(plk1, plk2, c='r')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.title('遗传算法求解的最优路径')
plt.show()

# 绘制迭代过程中最短路径长度的变化图
plot = plt.plot(ctlist, dislist)
plt.xlabel('迭代次数')
plt.ylabel('最短路径长度')
plt.title('迭代过程中最短路径长度变化')
plt.show()
~~~



