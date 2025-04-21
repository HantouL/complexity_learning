import matplotlib
matplotlib.use('TkAgg')

import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 导入 tqdm 用于进度条


def ba_graph(N, m0, m):
    # 初始化图
    G = nx.complete_graph(m0)
    # nx.draw(G)
    # plt.show()

    # 增长
    for node in tqdm(range(m0, N), desc="Building BA Graph"):
        # 引入一个新的节点
        G.add_node(node)
        # 将节点连到m个已存在的节点上
        degrees = dict(G.degree())  # 把每个节点的度找出来
        total_degrees = sum(degrees.values())  # 所有节点度的和
        # 算出每个节点被连接的概率
        nodes_list = list(G.nodes())
        # probabilities = [degrees[node]/total_degrees for node in nodes_list]
        if total_degrees == 0:
            # 如果 total_degrees 为 0，均匀随机选择节点
            probabilities = [1.0 / len(nodes_list) for _ in nodes_list]
        else:
            # 否则使用优先连接概率
            probabilities = [degrees[node] / total_degrees for node in nodes_list]

        # 开始连接
        selected_nodes = set()
        while len(selected_nodes)<m:
            chosen_node = random.choices(nodes_list, weights=probabilities, k=1)[0]
            if chosen_node != node and chosen_node not in selected_nodes:
                G.add_edge(node, chosen_node)
                selected_nodes.add(chosen_node)

    return G

# 计算度分布
# 返回度和对应的概率
def degree_destribution(G):
    degrees = [d for n,d in G.degree()]
    max_degree = max(degrees)
    # 每个度出现次数
    degree_cnts = np.bincount(degrees)
    # 计算度分布
    degree_prob = degree_cnts/len(degrees)
    return range(len(degree_prob)), degree_prob

# 图8-2 不同m的度分布
def show_diff_m():
    N = 300000
    m_values = [1,3,5,7]

    plt.figure(figsize=(8,6))
    for m in m_values:
        G = ba_graph(N, m, m)
        k, pk = degree_destribution(G)
        plt.scatter(k, pk, label=f'm_0 = {m}', s=10, alpha=0.6)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('不同 m_0 值时的 BA 模型度分布')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()



# 图8-3 不同N的度分布
def show_diff_N():
    N_values = [100000,150000,200000]
    m = 5
    m0=5

    plt.figure(figsize=(8,6))
    for N in N_values:
        G = ba_graph(N, m0, m)
        k, pk = degree_destribution(G)
        plt.scatter(k, Pk, label=f'N = {N}', s=10, alpha=0.6)

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