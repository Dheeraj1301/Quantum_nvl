"""Utilities for embedding graphs using a quantum-inspired kernel.

This module previously relied on Qiskit's :class:`~qiskit_machine_learning.kernels.QuantumKernel`.  To
avoid dependency conflicts we implement an alternative that optionally uses
PennyLane's kernel functions when available and falls back to a simple
classical radial basis function kernel otherwise.
"""

from typing import List
import networkx as nx

try:  # pragma: no cover - optional dependency
    import pennylane as qml
    _HAS_PENNYLANE = True
except Exception:  # pragma: no cover - pennylane not installed
    qml = None
    _HAS_PENNYLANE = False

import numpy as np
from sklearn.decomposition import PCA


class QuantumGraphEmbedder:
    """Embed graphs via a kernel matrix reduced with PCA."""

    def __init__(self, pca_components: int = 3) -> None:
        self.pca = PCA(n_components=pca_components)

    def _kernel_matrix(self, vectors: np.ndarray) -> np.ndarray:
        if _HAS_PENNYLANE and hasattr(qml.kernels, "projected_distance"):
            return qml.kernels.projected_distance(vectors, scale=0.5)
        sq_dists = np.sum((vectors[:, None, :] - vectors[None, :, :]) ** 2, axis=-1)
        return np.exp(-0.5 * sq_dists)

    def embed_graphs(self, graph_list: List["nx.Graph"]) -> np.ndarray:
        vectors = np.array([self._extract_features(g) for g in graph_list])
        kernel_matrix = self._kernel_matrix(vectors)
        return self.pca.fit_transform(kernel_matrix)

    def _extract_features(self, g) -> np.ndarray:
        return np.array([
            np.mean([g.degree(n) for n in g.nodes]),
            np.std([g.nodes[n].get("load", 1) for n in g.nodes]),
            np.mean([d.get("bandwidth", 1) for _, _, d in g.edges(data=True)]),
            len(g.nodes),
        ])
