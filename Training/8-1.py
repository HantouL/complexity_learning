import matplotlib

matplotlib.use('TkAgg')

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# 生成BA图的函数
def ba_graph(N, m0, m):
    # 初始化一个完全图，节点数为m0
    G = nx.complete_graph(m0)

    # 使用numpy数组存储度数，提高访问速度
    degrees = np.array([m0 - 1] * m0, dtype=float)
    total_degree = np.sum(degrees)
    # 预分配边列表以减少动态调整
    edges = list(G.edges())
    # 使用numpy的随机选择函数
    nodes = np.arange(m0)

    # 逐步添加节点并构建图
    for node in tqdm(range(m0, N), desc="构建BA图"):
        # 计算连接概率
        if total_degree == 0:
            probs = np.ones(len(nodes)) / len(nodes)
        else:
            probs = degrees / total_degree

        # 使用numpy随机选择m个节点
        targets = np.random.choice(nodes, size=m, replace=False, p=probs)

        # 添加边
        new_edges = [(node, target) for target in targets]
        edges.extend(new_edges)

        # 更新度数
        degrees = np.pad(degrees, (0, 1), mode='constant')  # 添加新节点
        degrees[targets] += 1  # 更新目标节点度数
        degrees[node] = m  # 新节点度数
        total_degree += 2 * m  # 每次添加m条边，总度数增加2m
        nodes = np.arange(node + 1)  # 更新节点列表

    # 创建最终图
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)

    return G


# 计算度分布
def degree_distribution(G):
    degrees = np.array([d for _, d in G.degree()])
    degree_counts = np.bincount(degrees)
    degree_prob = degree_counts / len(degrees)
    return np.arange(len(degree_prob)), degree_prob


# 不同m值的度分布
def show_diff_m():
    N = 30000  # 设置节点总数为300,00
    m_values = [1, 3, 5, 7]  # 不同m值
    markers = ['o', 's', 'd', '^']  # 圆圈、方块、菱形、三角形
    colors = ['green', 'yellow', 'blue', 'red']

    # 创建主图+插图显示缩放后的分布 P(k)/(2m^2)
    fig, ax = plt.subplots(figsize=(8, 6))
    inset_ax = fig.add_axes([0.5, 0.5, 0.3, 0.3])
    for i, m in enumerate(m_values):
        G = ba_graph(N, m, m)  # m0 = m
        k, pk = degree_distribution(G)
        ax.scatter(k, pk, label=f'm={m}', s=30, alpha=1.0, marker=markers[i], color=colors[i], edgecolors='none')

        pk_rescaled = pk / (2 * m ** 2)  # 缩放P(k)
        mask = k < 1000
        inset_ax.scatter(k[mask], pk_rescaled[mask], s=30, alpha=1.0, marker=markers[i], color=colors[i],
                         edgecolors='none')

    # 设置对数刻度
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$k$')
    ax.set_ylabel('$P(k)$')
    ax.set_title('different M')
    ax.legend()
    ax.grid(False)

    # 设置插图的对数刻度
    inset_ax.set_xscale('log')
    inset_ax.set_yscale('log')
    inset_ax.set_xlabel('$k$')
    inset_ax.set_ylabel('$P(k)/(2m^2)$')
    inset_ax.grid(False)
    plt.show()


# 不同N值的度分布和时间演化插图
def show_diff_N():
    N_values = [10000, 15000, 20000]
    m = 5
    m0 = 5
    markers = ['o', 's', 'd']  # 圆圈、方块、菱形
    colors = ['green', 'yellow', 'blue']

    # 创建主图
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, N in enumerate(N_values):
        G = ba_graph(N, m0, m)
        k, pk = degree_distribution(G)
        ax.scatter(k, pk, label=f'N={N}', s=30, alpha=1.0, marker=markers[i], color=colors[i], edgecolors='none')

    # 设置对数刻度
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$k$')
    ax.set_ylabel('$P(k)$')
    ax.set_title('different N')
    ax.legend()
    ax.grid(False)

    # 创建插图显示两个顶点的连通性时间演化
    inset_ax = fig.add_axes([0.2, 0.2, 0.3, 0.3])
    # 模拟时间演化（t1=5, t2=95）
    N_sim = 5000  # 模拟的节点数
    G = nx.complete_graph(m0)  # 初始化完全图
    degree_t1 = []  # t1=5加入的节点的度数
    degree_t2 = []  # t2=95加入的节点的度数

    # 逐步构建图并跟踪5号点和95号点的度数
    for t in range(m0, N_sim):
        degrees = np.array([d for _, d in G.degree()], dtype=float)
        total_degree = np.sum(degrees)
        probs = degrees / total_degree
        nodes = np.arange(len(degrees))
        targets = np.random.choice(nodes, size=m, replace=False, p=probs)
        G.add_node(t)
        G.add_edges_from([(t, target) for target in targets])

        if t >= 5:
            degree_t1.append(G.degree(5))  # 跟踪节点5的度数
        if t >= 95:
            degree_t2.append(G.degree(95))  # 跟踪节点95的度数

    # 绘制时间演化曲线，使用散点图以匹配目标图像
    time_steps_t1 = list(range(5, N_sim))
    inset_ax.scatter(time_steps_t1, degree_t1, color='black', s=30, alpha=0.7, marker='o', label='t1=5')
    # degree_t2对应的时间步长从t=95开始
    time_steps_t2 = list(range(95, N_sim))
    inset_ax.scatter(time_steps_t2, degree_t2, color='black', s=30, alpha=0.7, marker='s', label='t2=95')

    # 设置插图的对数刻度
    inset_ax.set_xscale('log')
    inset_ax.set_yscale('log')
    inset_ax.set_xlabel('$t$')
    inset_ax.set_ylabel('$k(t)$')
    inset_ax.legend()
    inset_ax.grid(False)
    plt.show()


# 调用函数
show_diff_m()
show_diff_N()
