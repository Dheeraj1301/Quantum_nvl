import networkx as nx
from quantumflow_ai.modules.q_nvlinkopt.nvlink_graph_optimizer import QAOANVLinkOptimizer
from quantumflow_ai.modules.q_nvlinkopt.qgnn_hybrid_optimizer import QAOA_GNN_Router


def simple_graph():
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    return g


def test_score_subgraph_range():
    g = simple_graph()
    opt = QAOANVLinkOptimizer(g)
    score = opt.score_subgraph(g)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_prepare_input_shapes():
    g = simple_graph()
    router = QAOA_GNN_Router()
    x, edge_index = router.prepare_input(g)
    assert x.shape[0] == g.number_of_nodes()
    assert edge_index.shape[1] == g.number_of_edges()
