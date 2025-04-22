import matplotlib

matplotlib.use('TkAgg')

import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def ba_graph(N, m0, m):
    # 初始化图
    G = nx.complete_graph(m0)

    # 使用numpy数组存储度数，提高访问速度
    degrees = np.array([m0 - 1] * m0, dtype=float)
    total_degree = np.sum(degrees)

    # 预分配边列表以减少动态调整
    edges = list(G.edges())

    # 使用numpy的随机选择函数替代random.choices
    nodes = np.arange(m0)

    for node in tqdm(range(m0, N), desc="Building BA Graph"):
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

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)

    return G


def degree_distribution(G):
    """计算度分布"""
    degrees = np.array([d for _, d in G.degree()])
    degree_counts = np.bincount(degrees)
    degree_prob = degree_counts / len(degrees)
    return np.arange(len(degree_prob)), degree_prob


def show_diff_m():
    N = 30000
    m_values = [1, 3, 5, 7]

    plt.figure(figsize=(8, 6))
    for m in m_values:
        G = ba_graph(N, m, m)
        k, pk = degree_distribution(G)
        plt.scatter(k, pk, label=f'm_0 = {m}', s=10, alpha=0.6)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('不同 m_0 值时的 BA 模型度分布')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


def show_diff_N():
    N_values = [10000, 15000, 20000]
    m = 5
    m0 = 5

    plt.figure(figsize=(8, 6))
    for N in N_values:
        G = ba_graph(N, m0, m)
        k, pk = degree_distribution(G)
        plt.scatter(k, pk, label=f'N = {N}', s=10, alpha=0.6)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('不同 N 值时的 BA 模型度分布')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()



show_diff_m()
show_diff_N()