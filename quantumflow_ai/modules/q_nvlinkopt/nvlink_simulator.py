# quantumflow_ai/modules/q_nvlinkopt/nvlink_simulator.py

import networkx as nx
import random

def generate_nvlink_graph(num_gpus=8, max_load=100):
    G = nx.Graph()
    for i in range(num_gpus):
        G.add_node(i, load=random.randint(10, max_load))
    
    # Simulate full NVLink mesh or hybrid topology
    for i in range(num_gpus):
        for j in range(i + 1, num_gpus):
            if random.random() < 0.5:
                G.add_edge(i, j, bandwidth=random.randint(1, 20))

    return G
