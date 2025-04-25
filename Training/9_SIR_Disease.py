import networkx as nx
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

er_graph = nx.erdos_renyi_graph(1000, 0.2) #1000个节点，连边概率0.2
ws_graph = nx.watts_strogatz_graph(1000,200,0.5) # 1000个节点，每个节点200个连边，重连概率0.2
ba_graph = nx.barabasi_albert_graph(1000,200) # 1000个节点，新节点会连200条边
