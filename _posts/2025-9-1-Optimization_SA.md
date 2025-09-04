---
title: 优化算法（1）模拟退火算法SA
tags: Optimization Algorithm
typora-root-url: ./..
---

模拟退火算法（Simulated Annealing, SA）是一种启发式搜索算法，它通过模拟物理中的退火过程来解决优化问题。这种算法能够跳出局部最优解，寻找全局最优解，适用于解决单目标、低复杂度优化问题。

<!--more-->

##### 1.原理介绍

模拟退火算法的核心原理是模拟物理中的退火过程，将问题的解状态视为物理系统的状态，目标函数值视为系统的能量。算法从初始温度开始，通过随机扰动当前解产生新解，并根据Metropolis准则决定是否接受新解。随着温度的逐渐降低，系统逐渐趋于稳定，最终在低温下达到全局最优或近似最优解。

模拟退火算法也是一种基于概率的优化算法，试图从一个初始解出发，逐步搜索其他可能解，以找到全局最优解。

##### 2.算法详解

###### 2.1 超参数

（1）初始温度：用于接受劣解的上限意愿。

（2）温度衰减系数：用于控制降温速度，一般在[0.88, 0.99]区间。越接近1，搜索越精细，但耗时越长，可选择自适应动态增大。

（2）迭代次数：每个温度下的迭代次数，让当前温度近似平衡，可自适应提前降温。

（4）终止温度：用于停止迭代的阈值。

###### 2.2 Metropolis准则

这是模拟退火能跳出局部最优解的核心。

如果没有这个准则，模型退化为贪心算法，找到一个解就结束，但是这个解可能是局部最优解，例如A点。而Metropolis准则用一定的概率决定要不要选择劣质解，也就是选择比当前解要差的解，从而跳到E点，最终达到全局最优解B点。在模拟退火中，解用状态描述，从当前解到新解被称为从当前状态到新状态。

![](/images/Optimization/2.png)

在温度T时，由当前状态i到新状态j。若$E_{j}<E_{i}$，则接受j作为新状态。否则，计算接受这个状态的概率，若p大于[0,1)之间的一个随机数，则仍然接受状态j作为当前最优解。否则保留状态i。其中$ k_{B} $ 可以调整整体接受程度。当 $ k_{B}  $越大，接受程度越高。（一般可以把 $ k_{B} $ 设置为1）

$$p=\left\{\begin{array}{l}
1, E_{j}<E_{i} \\
e^{-\frac{E_{j}-E_{i}}{k_{B} * T}}, E_{j} \geq E_{i}
\end{array}\right.$$

- 温度越高，指数曲线越平坦，即使$E_{j}-E_{i}$很大，P 仍接近 1 → 算法“敢”跳坑。
- 温度越低，指数曲线越陡峭，哪怕$E_{j}-E_{i}$很小，P 也接近 0 → 算法趋于保守，只接受改进。

###### 2.3 算法流程

（1）初始化：

- 随机生成初始解

- 设置初始温度

- 设置降温速率

- 设置终止温度

- 设置每个温度下的迭代次数

（2）迭代优化

- 在当前温度进行迭代
  - 从当前解生成一个邻近解
  - 计算当前解和邻近解的变化
  - 如果新解更优，接受新解
  - 如果新解较差，以一定概率接受新解（概率随温度降低而减小）
-  按照温度系数降低温度

（3）终止条件

- 当温度低于终止温度，算法停止
- 输出最优解

![](/images/Optimization/1.png)

##### 3.实际运用

###### 3.1 函数极值求解

求解函数 $f (x) = x・sin (10π・x) + 2.0$在区间 [-1, 3] 上的最小值。

算法思路：

（1）定义函数，由于限定范围，可以从范围最小值开始逐步向右到范围最大值，即初始解选择下限，邻解选择右边的点。

（2）定义初始温度，降温速率，终止温度，迭代次数。

（3）初始化后，进入循环，每次向右找到一个新解，判断新解和当前解的优劣，如果新解优，则代替当前解，继续判断新解和最优解优劣；如果当前解优，则根据Metropolis准则选择是否更新解。

（4）记录当前最优解，降温以更新温度，温度低于阈值，则退出循环，输出最优解、最优解列表。

~~~
# 案例一：f (x) = x・sin (10π・x) + 2.0在区间 [-1, 3] 上的最小值
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]  # 支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def function(x):
    return x * np.sin(10*np.pi*x) + 2

T_0 = 1000 # 初始温度
alpha = 0.8 # 降温速率
T_end = 1 # 终止温度
n = 100 # 迭代次数
bounds = [(1,3)] #  变量范围

# 模拟退火算法
def simulated_annealing(function, bounds, n, T_0, alpha, T_end):
    """ 
    参数:
    function: 目标函数
    bounds: 变量边界，格式为[(min, max)]
    n: 每个温度下的迭代次数
    T_0: 初始温度
    alpha: 降温系数
    T_end: 终止温度
    
    返回:
    best_solution: 找到的最优解
    best_score: 最优解对应的目标函数值
    scores: 迭代过程中的最优值记录
    """
    # 初始化解
    best_solution = bounds[0][0] + np.random.rand() * (bounds[0][1] - bounds[0][0])
    best_score = function(best_solution)
    
    current_solution = best_solution
    current_score = best_score
    
    current_T = T_0
    scores = [best_score]
    
    # 算法主循环
    while current_T > T_end:
        for _ in range(n):
            # 生成邻近解
            neighbor = current_solution + np.random.normal(0, 0.1)
            # 确保解在边界内
            neighbor = np.clip(neighbor, bounds[0][0], bounds[0][1])
            
            # 计算目标函数值
            neighbor_score = function(neighbor)
            
            # 检查是否为更优解
            if neighbor_score < current_score:
                current_solution = neighbor
                current_score = neighbor_score
                
                # 更新全局最优
                if neighbor_score < best_score:
                    best_solution = neighbor
                    best_score = neighbor_score
            else:
                # 计算接受概率
                accept_prob = np.exp((current_score - neighbor_score) / current_T)
                # 以一定概率接受较差解
                if np.random.rand() < accept_prob:
                    current_solution = neighbor
                    current_score = neighbor_score
        
        # 记录当前最优值
        scores.append(best_score)
        # 降温
        current_T = alpha * current_T
    
    return best_solution, best_score, scores

# 运行模拟退火算法
best_solution, best_score, scores = simulated_annealing(
    function, bounds, n, T_0, alpha, T_end)

# 输出结果
print(f"最优解: x = {best_solution:.4f}")
print(f"最优值: f(x) = {best_score:.4f}")

# 可视化结果
x = np.linspace(bounds[0][0], bounds[0][1], 1000)
y = function(x)

plt.figure(figsize=(12, 6))

# 绘制函数曲线
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', label='目标函数')
plt.scatter(best_solution, best_score, color='red', s=100, label='最优解')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('函数曲线与最优解')
plt.legend()

# 绘制收敛曲线
plt.subplot(1, 2, 2)
plt.plot(scores, 'g-')
plt.xlabel('迭代次数')
plt.ylabel('最优目标函数值')
plt.title('算法收敛曲线')

plt.show()
~~~

###### 3.2  旅行商问题

把城市数据转为csv。

~~~
import pandas as pd

# 清洗后的数据（城市, 东经, 北纬）
raw_lines = """\
北京,116.46,39.92
天津,117.20,39.13
上海,121.48,31.22
重庆,106.54,29.59
拉萨,91.11,29.97
乌鲁木齐,87.68,43.77
银川,106.27,38.47
呼和浩特,111.65,40.82
南宁,108.33,22.84
哈尔滨,126.63,45.75
长春,125.35,43.88
沈阳,123.38,41.80
石家庄,114.48,38.03
太原,112.53,37.87
西宁,101.74,36.56
济南,117.00,36.65
郑州,113.60,34.76
南京,118.78,32.04
合肥,117.27,31.86
杭州,120.19,30.26
福州,119.30,26.08
南昌,115.89,28.68
长沙,113.00,28.21
武汉,114.31,30.52
广州,113.23,23.16
台北,121.50,25.05
海口,110.35,20.02
兰州,103.73,36.03
西安,108.95,34.27
成都,104.06,30.67
贵阳,106.71,26.57
昆明,102.73,25.04
香港,114.10,22.20
澳门,113.33,22.13
"""

records = [line.split(',') for line in raw_lines.strip().splitlines()]
df = pd.DataFrame(records, columns=['city', 'lon', 'lat'])
df['lon'] = df['lon'].astype(float)
df['lat'] = df['lat'].astype(float)

# 保存 CSV
df.to_csv('city_coord.csv', index=False, encoding='utf_8_sig')
print(1)
~~~

使用模拟退火求解旅行商问题。

算法思路：

（1）旅行商问题要求不重复经过城市的情况下求最短的路径，那目标函数就是最短路径的总距离，距离计算方式选择haversine公式，用于计算经纬度数据距离。

（2）函数确定后，随机生成一条初始路径，之后选择产生邻解的方法，从a.随机选择两座城市交换，b.随机选择相邻两座城市进行交换，c.随机选择子路径城市逆转，d.随机选择子路径城市移位，选择一种方法，也可以四种方法混合使用。

（3）定义初始温度，降温速率，终止温度，迭代次数，城市数，距离矩阵。

（4）随机生成初始解，选择方法产生邻解，进行判断，判断结束后更新当前温度最优解，更新温度，如果温度低于阈值则停止迭代。

~~~
# 旅行商问题
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams["font.family"] = ["SimHei"]  # 支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

df = pd.read_csv('city_coord.csv', encoding='utf-8')

def haversine(lat1, lon1, lat2, lon2, radius=6371):
    # 将经纬度从度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # 计算经纬度差值
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    
    # 计算 Haversine 公式中的 a
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
    
    # 计算 c
    c = 2 * math.asin(math.sqrt(a))
    
    # 计算距离 d
    d = radius * c
    
    return d
    
dis = []
for i in range(len(df['lon'])):
    row = []
    for j in range(len(df['lat'])):
        h = haversine(df.iloc[i,2], df.iloc[i,1], df.iloc[j,2], df.iloc[j,1], 6371)
        row.append(round(h,2))
    dis.append(row)
    row = []

# 模拟退火算法
def function(route, dis):
    total_dis = 0
    for i in range(len(route)):
        city1 = route[i]
        city2 = route[(i + 1) % len(route)]
        total_dis += dis[city1][city2]
    return total_dis

# 多种邻域搜索策略
def generate_neighbor(route, method=None):
    route = route.copy()
    n = len(route)
    
    # 随机选择一种邻域操作
    if method is None:
        method = random.choice([1, 2, 3, 4])
    
    # 1. 随机交换两个城市
    if method == 1:
        i, j = random.sample(range(n), 2)
        route[i], route[j] = route[j], route[i]
    
    # 2. 相邻交换
    elif method == 2:
        i = random.randint(0, n-2)
        route[i], route[i+1] = route[i+1], route[i]
    
    # 3. 子路径逆转
    elif method == 3:
        i, j = random.sample(range(n), 2)
        if i > j:
            i, j = j, i
        route[i:j+1] = route[i:j+1][::-1]
    
    # 4. 子路径移位
    elif method == 4:
        i, j, k = random.sample(range(n), 3)
        if i > j:
            i, j = j, i
        # 提取子路径
        sub = route[i:j+1]
        # 删除子路径
        del route[i:j+1]
        # 插入子路径到新位置
        route[k:k] = sub
    
    return route
    
T_0 = 1000
alpha = 0.95
T_end = 0.001
n = 100
citys = len(dis)

def SA(citys, dis, T_0, alpha, T_end, n):
    # 产生初始解
    best_route = list(np.random.permutation(citys))
    best_dis = function(best_route, dis)

    current_route = best_route
    current_dis = best_dis

    current_T = T_0
    dis_history = [best_dis]

    while current_T > T_end:
        for _ in range(n):
            new_route = generate_neighbor(current_route,3)
            new_dis = function(new_route, dis)

            if new_dis < current_dis:
                current_route = new_route
                current_dis = new_dis

                if new_dis < best_dis:
                    best_route = new_route
                    best_dis = new_dis
            else:
                acceptance_prob = np.exp((current_dis - new_dis) / current_T)
                # 以一定概率接受较差解
                if np.random.rand() < acceptance_prob:
                    current_route = new_route
                    current_dis = new_dis

        dis_history.append(best_dis)
        current_T = alpha * current_T

    return best_route, best_dis, dis_history

best_route, best_dis, dis_history = SA(citys, dis, T_0, alpha, T_end, n)

print(best_route), print(best_dis)

# 可视化结果
plt.figure(figsize=(12, 6))

# 绘制最优路径
plt.subplot(1, 2, 1)
# 绘制城市
for i in range(citys):
    plt.scatter(df.iloc[i,1], df.iloc[i,2], color='blue')
    plt.text(df.iloc[i,1], df.iloc[i,2]+0.3, f'{i}', fontsize=12)

# 绘制路径
for i in range(citys):
    city1 = best_route[i]
    city2 = best_route[(i + 1) % citys]
    x1, y1 = df.iloc[city1,1], df.iloc[city1,2]
    x2, y2 = df.iloc[city2,1], df.iloc[city2,2]
    plt.plot([x1, x2], [y1, y2], 'r-')

plt.title(f'TSP最优路径 (距离: {best_dis:.2f})')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')

# 绘制收敛曲线
plt.subplot(1, 2, 2)
plt.plot(dis_history, 'g-')
plt.title('路径距离收敛曲线')
plt.xlabel('迭代次数')
plt.ylabel('路径总距离')

# plt.tight_layout()
plt.show()
~~~

![](/images/Optimization/3.png)

代码内含有四种产生邻解的方式，图中为第三种方法的结果，虽然每次结果不一样，但是也可以大致最优解的范围。

