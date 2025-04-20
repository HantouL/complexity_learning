import matplotlib
matplotlib.use('TkAgg')
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, binom


def GNL(N, L):
    G = nx.Graph()  # 生成个图
    G.add_nodes_from(range(N))  # 把节点添加进去
    nlist = list(G)
    edge_count = 0

    while edge_count < L:
        # 随机选一条边
        u = random.choice(nlist)
        v = random.choice(nlist)
        # 这条边非自环且该边在不在现存边集内
        if u == v or G.has_edge(u, v):
            continue
        else:
            G.add_edge(u, v)
            edge_count += 1
    return G


# g = GNL(20, 40)
#
# plt.figure(figsize=(8, 4))
# plt.subplot(121)
# nx.draw(g, pos=nx.circular_layout(g), node_size=300, node_color="red", with_labels=True)
# plt.title("G(N,L)")
# plt.show()


# 计算度分布
def get_pdf(G, kmin, kmax):
    k = list(range(kmin, kmax+1))
    N = len(G.nodes())

    Pk = []
    for ki in k:
        c = 0
        for i in G.nodes():
            if G.degree(i) == ki:
                c += 1
        Pk.append(c/N)
    return k, Pk

samples = 100
N = 1000
kmin, kmax = 0, 40
avk = 20  # 期望的平均度
L = int(N * avk / 2)

s = np.zeros(kmax-kmin+1)  # 累加度分布
for i in range(samples):
    ER = GNL(N,L)
    x,y = get_pdf(ER, kmin, kmax)
    s += np.array(y)

# 平均度概率
pk = s / samples

# 理论分布（泊松 & 二项）
k_vals = np.arange(kmin, kmax+1)
poisson_vals = poisson.pmf(k_vals, mu=avk)
binomial_vals = binom.pmf(k_vals, N-1, avk/(N-1))  # 近似ER G(N,p)模型

# 画图
plt.figure(figsize=(7, 5))
plt.plot(k_vals, pk, 'ro', label='实验值')
plt.plot(k_vals, poisson_vals, 'k-', label='泊松分布', alpha=0.7)
plt.plot(k_vals, binomial_vals, 'b--', label='二项分布', alpha=0.7)

plt.xlabel("$k$")
plt.ylabel("$P(k)$")
plt.title("ER G(N,L)图的度分布")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()