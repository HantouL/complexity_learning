# 第一讲 复杂网络基础

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

import networkx as nx

# G = nx.Graph()
# G.add_nodes_from([1, 2, 3, 4])
# G.add_edges_from([(1,2), (1,3), (2,3), (2,4)])

# nx.draw(G, node_size=500, with_labels=True)
# plt.show()  # 添加这一行以显示图形

# # 获取邻接矩阵
# As = nx.adjacency_matrix(G)
# print('G的邻接矩阵如下')
# print(As)
#
# # 转化成二维数组形式
# A = As.todense()
# print('G的二维数组形式的矩阵如下')
# print(A)
#
# # 已知邻接矩阵 创建图
# B = np.array([[0,1,1],[1,0,1],[1,1,0]])
# G1 = nx.from_numpy_array(B)
# nx.draw(G1, node_size=500, with_labels=True)
# plt.show()
#
# # 加权图
# C = np.array([[0,1,3.0],[1,2,7.5],[0,2,1.5]])
# G2 = nx.from_numpy_array(C)
# nx.draw(G2, node_size=500, with_labels=True)
# plt.show()
#
# # 有向图
# G3 = nx.DiGraph()
# G3.add_nodes_from([1,2,3,4])
# G3.add_edges_from([(1,2),(1,3),(2,3),(3,4)])
# nx.draw(G3, node_size=500, with_labels=True)
# plt.show()

# # 获取网络G的度
# d = dict(nx.degree(G))
# print('获取网络G的度')
# print(d)
# print('网络G的平均度 ', sum(d.values()) / len(G.nodes()))
#
# # 获取度分布
# print(nx.degree_histogram(G))
#
# # 绘制度分布直方图
# x = list(range(max(d.values())+1))
# y = [i/len(G.nodes) for i in nx.degree_histogram(G)]
# plt.bar(x,y,width=0.5,color='red')
# plt.xlabel("$k$",fontsize=14)
# plt.ylabel("$p_k$",fontsize=14)
# plt.xlim([0,4])
# plt.show()

# 网络路径和距离
G = nx.Graph()
G.add_nodes_from(range(1,5))
G.add_edges_from([(1,2),(2,3),(2,5),(3,4),(4,5)])
nx.draw(G, node_size=500,with_labels=True)
plt.show()

path = nx.shortest_path(G, source=1, target=4)
print(path)
#展示两个节点之间所有最短路径
lists = list(nx.all_shortest_paths(G, source=1, target=4))
print(lists)

# 求两个节点的最短路径长度
print(nx.shortest_path_length(G, source=1, target=4))
# 求整个网络平均距离
print(nx.average_shortest_path_length(G))

# 连通性
Ga = nx.Graph()
Ga.add_nodes_from(range(1,8))
Ga.add_edges_from([(1,2),(1,3),(2,3),(4,7),(5,6),(5,7),(6,7)])
nx.draw(Ga, node_size=500,with_labels=True)
plt.show()
print(nx.is_connected(Ga))

# 集聚系数
print(nx.average_clustering(G))