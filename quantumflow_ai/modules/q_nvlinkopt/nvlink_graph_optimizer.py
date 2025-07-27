# quantumflow_ai/modules/q_nvlinkopt/nvlink_graph_optimizer.py

from __future__ import annotations
import networkx as nx
import numpy as np
import pennylane as qml
from quantumflow_ai.core.quantum_backend import get_quantum_device
from quantumflow_ai.core.logger import get_logger

logger = get_logger("QAOA_NVLink_Optimizer")

class QAOANVLinkOptimizer:
    def __init__(self, nvlink_graph: nx.Graph, depth: int = 2):
        self.graph = nvlink_graph
        self.depth = depth
        self.num_nodes = self.graph.number_of_nodes()

        # Map arbitrary node labels to a contiguous wire range starting at 0.
        # PennyLane devices expect wire labels to exist on the device, so we
        # create a mapping that translates graph node identifiers to
        # sequential indices used as wire labels in the circuit.
        self.nodes = list(self.graph.nodes())
        self.wire_map = {node: idx for idx, node in enumerate(self.nodes)}

        self.dev = get_quantum_device(wires=self.num_nodes)

    def _cost_hamiltonian(self):
        """Cost function: minimize total edge conflicts (congestion)"""
        edges = [(self.wire_map[i], self.wire_map[j]) for i, j in self.graph.edges()]
        def cost_fn(z):
            return sum((1 - z[i]*z[j])/2 for i, j in edges)
        return cost_fn

    def optimize(self):
        """Run QAOA optimization"""
        from pennylane.optimize import AdamOptimizer

        def qaoa_layer(gamma, beta):
            for i, j in self.graph.edges():
                iw, jw = self.wire_map[i], self.wire_map[j]
                qml.CNOT(wires=[iw, jw])
                qml.RZ(-gamma, wires=jw)
                qml.CNOT(wires=[iw, jw])
            for i in range(self.num_nodes):
                qml.RX(2 * beta, wires=i)

        def circuit(params):
            gamma, beta = params
            for i in range(self.num_nodes):
                qml.Hadamard(wires=i)
            for _ in range(self.depth):
                qaoa_layer(gamma, beta)
            return qml.expval(qml.Hermitian(np.eye(2**self.num_nodes), wires=range(self.num_nodes)))

        qnode = qml.QNode(circuit, self.dev)
        opt = AdamOptimizer(0.1)
        params = np.random.random(2)

        for _ in range(25):
            params = opt.step(lambda v: -qnode(v), params)

        logger.info(f"QAOA optimized params: {params}")
        return self._extract_plan(params)

    def _extract_plan(self, params):
        """Convert optimized angles into coloring/partitioning"""
        gamma, beta = params
        colors = {
            node: int(np.round(np.sin(gamma + idx * beta))) % 3
            for node, idx in self.wire_map.items()
        }
        return colors

    def score_subgraph(self, subgraph: nx.Graph) -> float:
        """Return a simple congestion score for ``subgraph`` using QAOA.

        The method runs a short QAOA optimization on the provided ``subgraph``
        and evaluates the resulting colouring by counting edge conflicts.  A
        higher score indicates fewer conflicts and therefore a better mapping.
        """

        if subgraph.number_of_nodes() == 0:
            return 0.0

        # Reuse the optimizer logic on the subgraph
        sub_opt = QAOANVLinkOptimizer(subgraph, depth=self.depth)
        plan = sub_opt.optimize()

        conflicts = sum(1 for u, v in subgraph.edges() if plan[u] == plan[v])
        max_conflicts = subgraph.number_of_edges() or 1

        # Normalized score in [0, 1], where 1.0 means no conflicts
        return 1.0 - (conflicts / max_conflicts)
