---
title: 优化算法（3）粒子群算法PSO
tags: Optimization Algorithm
typora-root-url: ./..
---

粒子群算法（Particle Swarm Optimization，PSO）是一种基于群体智能的优化算法，通过模拟鸟群中个体之间的信息交流和协作来寻找最优解。

<!--more-->

##### 1.粒子群算法简介

粒子群算法将每个潜在解视为一个粒子，粒子在解空间中飞行，通过跟踪个体最优解（pbest）和群体最优解（gbest）来调整飞行方向和速度。每个粒子都有自己的位置和速度，位置表示当前解，速度表示位置的变化量。粒子通过不断更新自己的位置和速度，逐渐逼近最优解。

##### 2.粒子群算法详解

###### 2.1 超参数

（1）粒子种群规模：粒子群中粒子的数量。粒子数目越大，算法搜索的空间范围就越大，也就更容易发现全局最优解，算法运行的时间也越长。

（2）初始位置、速度：每个粒子的初始位置、初始飞行速度。

（3）惯性权重：控制粒子前一速度的影响程度。当惯性权重值较大时，全局寻优能力较强，局部寻优能力较弱；当惯性权重值较小时，全局寻优能力较弱，局部寻优能力较强。

（4）加速常数$c_1$和$c_2$：分别控制粒子向个体最优解和全局最优解移动的速度，分别决定粒子个体经验和群体经验对粒子运行轨迹的影响。

（5）粒子的最大速度：限制粒子飞行速度。过小容易陷入局部最优，过大会略过最优值。

（6）迭代次数：算法运行的最大迭代次数。

（7）邻域拓扑结构：定义粒子之间的信息交流方式。

###### 2.2 更新公式

假设粒子群有N个粒子，每个粒子的位置为$x_i$，速度为$v_i$，个体最优解为pbest，全局最优解为gbest。

速度更新公式为：

$$ v_i^{(t+1)} =v_i^{(t)} + c_1 r_1 (pbest_i - x_i^{(t)}) + c_2 r_2 (gbest_i - x_i^{(t)}) $$

其中，$r_1$和$r_2$是0-1之间的随机数，用于增加算法的随机性和全局搜索能力。

位置更新公式为：

$$ x_i^{(t+1)}=x_i^{(t)} + v_i^{(t+1)} $$

###### 2.3 算法流程

![](/images/Optimization/5.png)

##### 3.案例

###### 3.1 函数求解

f (x) = x・sin (10π・x) + 2.0在区间 [-1, 3] 上的最小值

~~~
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["font.family"] = ["SimHei"]  # 支持中文
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def f(x):
    return x * np.sin(10 * np.pi * x) + 2.0


# 粒子群算法参数
num_particles = 30
num_iterations = 100
c1 = 2.0
c2 = 2.0
w = 0.7
v_max = 0.4
x_min = -1
x_max = 3

# 初始化粒子位置和速度
np.random.seed(42)
positions = np.random.uniform(x_min, x_max, num_particles)
velocities = np.random.uniform(-v_max, v_max, num_particles)

pbest_fitness = f(positions)
pbest_positions = positions.copy()
gbest_idx = np.argmin(pbest_fitness)
gbest_position = positions[gbest_idx]
gbest_fitness = pbest_fitness[gbest_idx]

# 存储优化过程数据
history_positions = [positions.copy()]  # 存储每代粒子x坐标
# 每代粒子的y值，都通过目标函数计算后存储
history_fitness = [f(positions).copy()]
history_gbest_pos = [gbest_position]
history_gbest_fit = [gbest_fitness]

# 粒子群优化过程
for iteration in range(num_iterations):
    # 1. 更新速度
    r1 = np.random.rand(num_particles)
    r2 = np.random.rand(num_particles)
    velocities = w * velocities + c1 * r1 * (pbest_positions - positions) + c2 * r2 * (gbest_position - positions)
    velocities = np.clip(velocities, -v_max, v_max)

    # 2. 更新位置
    positions += velocities
    positions = np.clip(positions, x_min, x_max)

    # 3. 位置更新后，重新计算y值
    current_fitness = f(positions) 

    # 4. 更新个体最优
    better_mask = current_fitness < pbest_fitness
    pbest_positions[better_mask] = positions[better_mask]
    pbest_fitness[better_mask] = current_fitness[better_mask]

    # 5. 更新全局最优
    global_best_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[global_best_idx]
    gbest_fitness = pbest_fitness[global_best_idx]

    # 6. 存储数据
    history_positions.append(positions.copy())
    history_fitness.append(current_fitness.copy())  # 存储新计算的y值
    history_gbest_pos.append(gbest_position)
    history_gbest_fit.append(gbest_fitness)

# 绘制目标函数完整曲线
x_full = np.linspace(x_min, x_max, 1000)
y_full = f(x_full)

# 创建画布与子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 子图1：目标函数曲线 + 粒子位置（确保粒子落在曲线上）
ax1.plot(x_full, y_full, 'b-', linewidth=1.5, label='目标函数 f(x)')
# 初始化散点（后续通过update函数更新）
scatter_pop = ax1.scatter([], [], c='red', s=30, alpha=0.6, label='当前粒子位置')
scatter_best = ax1.scatter([], [], c='orange', s=100, marker='*', label='全局最优')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('粒子群位置分布（迭代过程）', fontsize=14)
ax1.set_xlim(x_min - 0.1, x_max + 0.1)
ax1.set_ylim(np.min(y_full) - 0.5, np.max(y_full) + 0.5)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：迭代次数 vs 全局最优值
line_best = ax2.plot([], [], 'g-', linewidth=2, label='全局最优值')[0]
ax2.set_xlabel('迭代次数', fontsize=12)
ax2.set_ylabel('f(x) 最小值', fontsize=12)
ax2.set_title('迭代过程中全局最优值变化', fontsize=14)
ax2.set_xlim(0, num_iterations)
ax2.set_ylim(np.min(history_gbest_fit) - 0.2, np.max(history_gbest_fit) + 0.2)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 文本标注（显示迭代信息）
text_iter = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def update(frame):
    """动画更新函数（确保粒子x与y严格对应）"""
    # 1. 获取当前帧的粒子x和y（均来自历史存储，y = f(x)）
    current_x = history_positions[frame]
    current_y = history_fitness[frame]  # 与current_x一一对应，来自目标函数
    # 更新粒子散点：x和y严格匹配，确保落在曲线上
    scatter_pop.set_offsets(np.c_[current_x, current_y])

    # 2. 更新全局最优散点
    best_x = history_gbest_pos[frame]
    best_y = history_gbest_fit[frame]
    scatter_best.set_offsets([[best_x, best_y]])

    # 3. 更新全局最优值曲线
    line_best.set_data(list(range(frame + 1)), history_gbest_fit[:frame + 1])

    # 4. 更新文本标注
    text_iter.set_text(f'迭代次数: {frame + 1}\n当前最优值: {best_y:.6f}\n最优x: {best_x:.6f}')

    return scatter_pop, scatter_best, line_best, text_iter


# 创建并显示动画
ani = FuncAnimation(fig, update, frames=num_iterations, interval=500,
                    blit=True, repeat=False)
plt.tight_layout()
plt.show()

# 输出最终结果
print(f"优化完成！最优位置: {gbest_position:.6f}, 最优适应度: {gbest_fitness:.6f}")
~~~

###### 3.2 TSP旅行商问题

~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# Haversine距离计算函数（计算地球表面两点间的距离，单位：千米）
def haversine_distance(lon1, lat1, lon2, lat2):
    # 将角度转换为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球半径，单位千米
    return c * r


# 计算路径总距离
def total_distance(path, distance_matrix):
    distance = 0
    num_cities = len(path)
    for i in range(num_cities):
        from_city = path[i]
        to_city = path[(i + 1) % num_cities]  # 最后一个城市回到起点
        distance += distance_matrix[from_city, to_city]
    return distance


# 初始化粒子群
def initialize_particles(num_particles, num_cities):
    particles = []
    for _ in range(num_particles):
        # 每个粒子是一个随机排列的城市索引
        particle = np.random.permutation(num_cities)
        particles.append(particle)
    return np.array(particles)


# 粒子群优化算法解决TSP
def pso_tsp(df, num_particles=30, num_iterations=100, w=0.7, c1=1.5, c2=1.5):
    # 提取城市数据
    cities = df['city'].values
    lons = df['lon'].values
    lats = df['lat'].values
    num_cities = len(cities)

    # 预计算距离矩阵
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i, j] = haversine_distance(
                    lons[i], lats[i], lons[j], lats[j]
                )

    # 初始化粒子群
    particles = initialize_particles(num_particles, num_cities)

    # 初始化个体最优和全局最优
    pbest_positions = particles.copy()
    pbest_distances = np.array([total_distance(p, distance_matrix) for p in particles])
    gbest_idx = np.argmin(pbest_distances)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_distance = pbest_distances[gbest_idx]

    # 存储优化过程用于可视化
    history_best_paths = [gbest_position.copy()]
    history_best_distances = [gbest_distance]
    history_all_paths = [particles.copy()]

    # PSO主循环
    for iteration in range(num_iterations):
        # 遍历每个粒子
        for i in range(num_particles):
            # 粒子当前路径
            current_path = particles[i].copy()

            # 生成新路径（基于粒子群优化的思想，通过交叉操作实现）
            # 1. 与个体最优交叉
            r1 = np.random.random()
            if r1 < c1 * np.random.random():
                # 随机选择交叉点
                a, b = np.random.choice(num_cities, 2, replace=False)
                if a > b:
                    a, b = b, a
                # 从个体最优中获取子路径
                sub_path = pbest_positions[i][a:b + 1]
                # 构建新路径
                new_path = []
                ptr = (b + 1) % num_cities
                for j in range(num_cities):
                    pos = (ptr + j) % num_cities
                    if current_path[pos] not in sub_path:
                        new_path.append(current_path[pos])
                    if j == a:
                        new_path.extend(sub_path)
                current_path = np.array(new_path[:num_cities])

            # 2. 与全局最优交叉
            r2 = np.random.random()
            if r2 < c2 * np.random.random():
                a, b = np.random.choice(num_cities, 2, replace=False)
                if a > b:
                    a, b = b, a
                sub_path = gbest_position[a:b + 1]
                new_path = []
                ptr = (b + 1) % num_cities
                for j in range(num_cities):
                    pos = (ptr + j) % num_cities
                    if current_path[pos] not in sub_path:
                        new_path.append(current_path[pos])
                    if j == a:
                        new_path.extend(sub_path)
                current_path = np.array(new_path[:num_cities])

            # 3. 随机扰动（模拟惯性权重）
            if np.random.random() < w:
                # 随机交换两个城市
                a, b = np.random.choice(num_cities, 2, replace=False)
                current_path[a], current_path[b] = current_path[b], current_path[a]

            # 更新粒子位置
            particles[i] = current_path

            # 计算当前路径距离
            current_distance = total_distance(current_path, distance_matrix)

            # 更新个体最优
            if current_distance < pbest_distances[i]:
                pbest_positions[i] = current_path.copy()
                pbest_distances[i] = current_distance

                # 更新全局最优
                if current_distance < gbest_distance:
                    gbest_position = current_path.copy()
                    gbest_distance = current_distance

        # 记录历史数据
        history_best_paths.append(gbest_position.copy())
        history_best_distances.append(gbest_distance)
        history_all_paths.append(particles.copy())

        # 打印进度
        if (iteration + 1) % 10 == 0:
            print(f"迭代 {iteration + 1}/{num_iterations}, 最优距离: {gbest_distance:.2f} 千米")

    return {
        'best_path': gbest_position,
        'best_distance': gbest_distance,
        'cities': cities,
        'lons': lons,
        'lats': lats,
        'history_best_paths': history_best_paths,
        'history_best_distances': history_best_distances,
        'history_all_paths': history_all_paths
    }


# 可视化TSP优化过程
def visualize_tsp(result):
    cities = result['cities']
    lons = result['lons']
    lats = result['lats']
    history_best_paths = result['history_best_paths']
    history_best_distances = result['history_best_distances']
    history_all_paths = result['history_all_paths']
    num_iterations = len(history_best_paths) - 1

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 城市位置散点
    ax1.scatter(lons, lats, c='blue', s=50, alpha=0.6, label='城市')
    for i, city in enumerate(cities):
        ax1.text(lons[i], lats[i], city, fontsize=9)

    # 初始化路径线
    best_path_line, = ax1.plot([], [], 'r-', linewidth=2, label='最优路径')
    all_paths_lines = [ax1.plot([], [], 'gray', linewidth=0.5, alpha=0.3)[0] for _ in history_all_paths[0]]

    ax1.set_xlabel('经度')
    ax1.set_ylabel('纬度')
    ax1.set_title('TSP路径优化过程')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 最优距离曲线
    distance_line, = ax2.plot([], [], 'g-', linewidth=2, label='最优距离')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('总距离（千米）')
    ax2.set_title('最优路径距离变化')
    ax2.set_xlim(0, num_iterations)
    ax2.set_ylim(min(history_best_distances) * 0.9, max(history_best_distances) * 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 文本标注
    text_iter = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                         fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update(frame):
        # 更新最优路径
        best_path = history_best_paths[frame]
        best_lons = [lons[i] for i in best_path] + [lons[best_path[0]]]
        best_lats = [lats[i] for i in best_path] + [lats[best_path[0]]]
        best_path_line.set_data(best_lons, best_lats)

        # 更新所有粒子的路径
        all_paths = history_all_paths[frame]
        for i, line in enumerate(all_paths_lines[:len(all_paths)]):
            path = all_paths[i]
            path_lons = [lons[j] for j in path] + [lons[path[0]]]
            path_lats = [lats[j] for j in path] + [lats[path[0]]]
            line.set_data(path_lons, path_lats)

        # 更新距离曲线
        distance_line.set_data(range(frame + 1), history_best_distances[:frame + 1])

        # 更新文本标注
        text_iter.set_text(f'迭代次数: {frame}\n最优距离: {history_best_distances[frame]:.2f} 千米')

        return [best_path_line] + all_paths_lines + [distance_line, text_iter]

    # 创建动画
    ani = FuncAnimation(fig, update, frames=num_iterations, interval=300,
                        blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

    return ani

df = pd.read_csv('city_coord.csv', encoding='utf-8')
print(f"成功读取 {len(df)} 个城市的数据")

# 运行PSO求解TSP
result = pso_tsp(df, num_particles=100, num_iterations=5000)

# 输出结果
print("\n优化完成！")
print(f"最优路径总距离: {result['best_distance']:.2f} 千米")
print("最优路径:")
best_path_cities = [result['cities'][i] for i in result['best_path']]
print(" -> ".join(best_path_cities) + " -> " + best_path_cities[0])

# 可视化优化过程
visualize_tsp(result)
~~~

