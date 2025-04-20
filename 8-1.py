import matplotlib
matplotlib.use('TkAgg')
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 导入 tqdm 用于进度条


def ba_graph(N, m0, m):
    # 初始化图
    G = nx.Graph()
    G.add_nodes_from(range(m0))
    for i in range(m0):
        for j in range (i+1, m0):
            G.add_edge(i, j)

    # 增长
    for node in range(m0, N):
        # 引入一个新的节点
        G.add_node(node)
        # 将节点连到m个已存在的节点上
        degrees = dict(G.degree)  # 把每个节点的度找出来
        total_degrees = sum(degrees.values())  # 所有节点度的和

        # 算出每个节点被连接的概率
        nodes_list = list(G.nodes())
        probabilities = [degrees[node]/total_degrees for node in nodes_list]

        # 开始连接
        selected_nodes = set()
        while len(selected_nodes)<m:
            chosen_node = random.choices(nodes_list, weights=probabilities, k=1)[0]
            if chosen_node != node and chosen_node not in selected_nodes:
                G.add_edge(node, chosen_node)

    return G


# 参数设置
N = 1000
K = 10
p_values = np.logspace(-4, 0, 14)  # p 从 0.0001 到 1，对数分布，参考图上有 14 个点
num_round = 20  # 每个数据 20 次取平均值

C_values = []
L_values = []
C0_trials = []
L0_trials = []

# 计算 p=0 时的基准值 C(0) 和 L(0)
print("计算 p=0 时的基准值 C(0) 和 L(0)...")
for _ in tqdm(range(num_round), desc="p=0 trials"):
    G0 = ws_smallworld(N, K, 0)
    C0_trials.append(nx.average_clustering(G0))
    L0_trials.append(nx.average_shortest_path_length(G0))
C0 = np.mean(C0_trials)
L0 = np.mean(L0_trials)
print(f"C(0) = {C0:.4f}, L(0) = {L0:.4f}")

# 对每个 p 计算 C(p) 和 L(p)
print("对每个 p 计算 C(p) 和 L(p)...")
for p in tqdm(p_values, desc="p values"):
    C_trials = []
    L_trials = []
    for _ in range(num_round):
        G = ws_smallworld(N, K, p)
        C_trials.append(nx.average_clustering(G))
        L_trials.append(nx.average_shortest_path_length(G))
    C_values.append(np.mean(C_trials) / C0)
    L_values.append(np.mean(L_trials) / L0)

# 绘制图形
plt.figure(figsize=(8, 6))
plt.scatter(p_values, C_values, marker='s', facecolors='none', edgecolors='black', label=r'$C(p)/C(0)$')
plt.scatter(p_values, L_values, marker='o', color='black', label=r'$L(p)/L(0)$')
plt.xscale('log')  # 横轴对数刻度
plt.xlabel(r'$p$', fontsize=12)
plt.ylabel(r'$C(p)/C(0)$ and $L(p)/L(0)$', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()