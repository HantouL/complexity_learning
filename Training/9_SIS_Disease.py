import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

er_graph = nx.erdos_renyi_graph(1000, 0.2) #1000个节点，连边概率0.2
ws_graph = nx.watts_strogatz_graph(1000,200,0.5) # 1000个节点，每个节点200个连边，重连概率0.2
ba_graph = nx.barabasi_albert_graph(1000,200) # 1000个节点，新节点会连200条边


# SIS 感染模型
def simu_sis(graph, beta, gamma, steps):
    # 初始化感染状态
    status = {node: 0 for node in graph.nodes()}
    status[0] = 1  # 指定0号病人

    # SIR人数
    total_nodes = graph.number_of_nodes()
    infected_cnt = [sum(1 for node, state in status.items() if state == 1)]
    susceptible_cnt = [total_nodes - infected_cnt[0]]

    # 开始模拟传播轮次
    for t in tqdm(range(steps), desc="疫情传播"):
        now_status = status.copy()

        for node in graph.nodes():
            if status[node] == 1:  # 感染节点
                for neighbor in graph.neighbors(node):
                    if status[neighbor] == 0 and random.random() < beta :
                        now_status[neighbor] = 1
                if random.random() < gamma:
                    now_status[node] = 0

        status = now_status
        infected = sum(1 for node, state in status.items() if state == 1)
        susceptible = total_nodes - infected

        infected_cnt.append(infected)
        susceptible_cnt.append(susceptible)

    return infected_cnt, susceptible_cnt

# 模拟参数
beta = 0.001
gamma = 0.005
steps = 1000

# 在三种图上运行 SI 模型
infected_er, susceptible_er = simu_sis(er_graph, beta, gamma, steps)
infected_ws, susceptible_ws = simu_sis(ws_graph, beta, gamma, steps)
infected_ba, susceptible_ba = simu_sis(ba_graph, beta, gamma, steps)

# 绘制结果
plt.figure(figsize=(10, 6))

# ER
plt.plot(infected_er, label="ER - Infected", color="red", linestyle="-")
plt.plot(susceptible_er, label="ER - Susceptible", color="blue", linestyle="--")
plt.xlabel("Time Step")
plt.ylabel("Number of Nodes")
plt.title("SIS Model: on ER Graphs")
plt.legend()
plt.grid(True)
plt.show()


# WS
plt.plot(infected_ws, label="WS - Infected", color="red", linestyle="-")
plt.plot(susceptible_ws, label="WS - Susceptible", color="blue", linestyle="--")
plt.xlabel("Time Step")
plt.ylabel("Number of Nodes")
plt.title("SIS Model: on WS Graphs")
plt.legend()
plt.grid(True)
plt.show()

# BA
plt.plot(infected_ba, label="BA - Infected", color="red", linestyle="-")
plt.plot(susceptible_ba, label="BA - Susceptible", color="blue", linestyle="--")
plt.xlabel("Time Step")
plt.ylabel("Number of Nodes")
plt.title("SIS Model: on BA Graphs")
plt.legend()
plt.grid(True)
plt.show()



