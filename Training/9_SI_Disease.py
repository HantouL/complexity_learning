import networkx as nx
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

er_graph = nx.erdos_renyi_graph(1000, 0.2) #1000个节点，连边概率0.2
ws_graph = nx.watts_strogatz_graph(1000,200,0.5) # 1000个节点，每个节点200个连边，重连概率0.2
ba_graph = nx.barabasi_albert_graph(1000,200) # 1000个节点，新节点会连200条边

# SI 感染模型
def simu_si(graph, beta, steps):
    # 初始化感染状态
    status = {node: 0 for node in graph.nodes()}
    status[0] = 1  # 指定0号病人

    # 感染者人数
    total_nodes = graph.number_of_nodes()
    infected_cnt = [sum(status.values())]
    susceptible_cnt = [total_nodes - infected_cnt[0]]

    # 开始模拟传播轮次
    for t in tqdm(range(steps), desc="疫情传播"):
        now_status = status.copy()
        for node in graph.nodes():
            if status[node] == 1:  # 感染节点
                for neighbor in graph.neighbors(node):
                    if status[neighbor] == 0 and random.random() < beta:
                        now_status[neighbor] = 1

        status = now_status
        infected_cnt.append(sum(status.values()))
        susceptible_cnt.append(total_nodes - sum(status.values()))
    return infected_cnt, susceptible_cnt

# 模拟参数
beta = 0.0001
steps = 1000

# 在三种图上运行 SI 模型
infected_er, susceptible_er = simu_si(er_graph, beta, steps)
infected_ws, susceptible_ws = simu_si(ws_graph, beta, steps)
infected_ba, susceptible_ba = simu_si(ba_graph, beta, steps)

# 绘制结果
plt.figure(figsize=(10, 6))
# 感染者曲线
plt.plot(infected_er, label="ER - Infected", color="blue", linestyle="-")
plt.plot(infected_ws, label="WS - Infected", color="green", linestyle="-")
plt.plot(infected_ba, label="BA - Infected", color="red", linestyle="-")
# 易感者曲线
plt.plot(susceptible_er, label="ER - Susceptible", color="blue", linestyle="--")
plt.plot(susceptible_ws, label="WS - Susceptible", color="green", linestyle="--")
plt.plot(susceptible_ba, label="BA - Susceptible", color="red", linestyle="--")

plt.xlabel("Time Step")
plt.ylabel("Number of Nodes")
plt.title("SI Model: Infected and Susceptible Nodes on Different Graphs")
plt.legend()
plt.grid(True)
plt.show()