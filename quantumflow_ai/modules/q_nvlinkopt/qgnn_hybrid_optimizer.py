import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
from modules.q_nvlinkopt.nvlink_graph_optimizer import QAOANVLinkOptimizer

class QAOA_GNN_Router(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, output_dim=2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def prepare_input(self, graph: nx.Graph):
        # Step 1: Run QAOA on local cliques or clusters
        qaoa_scores = {}
        optimizer = QAOANVLinkOptimizer(graph)
        for node in graph.nodes():
            local_cluster = graph.subgraph(list(graph.neighbors(node)) + [node])
            score = optimizer.score_subgraph(local_cluster)
            qaoa_scores[node] = score

        # Step 2: Extract node features: [load, degree, qaoa_score]
        features = []
        for node in graph.nodes():
            load = graph.nodes[node].get("load", 1)
            degree = graph.degree[node]
            qscore = qaoa_scores.get(node, 0.0)
            features.append([load, degree, qscore])

        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
        return x, edge_index
