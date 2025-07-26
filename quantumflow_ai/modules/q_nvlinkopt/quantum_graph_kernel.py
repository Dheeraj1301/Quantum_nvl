from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.utils import QuantumInstance
from qiskit import BasicAer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

class QuantumGraphEmbedder:
    def __init__(self, pca_components=3):
        self.q_instance = QuantumInstance(BasicAer.get_backend("qasm_simulator"))
        self.feature_map = ZZFeatureMap(feature_dimension=4, reps=2)
        self.qkernel = QuantumKernel(feature_map=self.feature_map, quantum_instance=self.q_instance)
        self.pca = PCA(n_components=pca_components)

    def embed_graphs(self, graph_list):
        vectors = [self._extract_features(g) for g in graph_list]
        kernel_matrix = self.qkernel.evaluate(x_vec=vectors)
        return self.pca.fit_transform(kernel_matrix)

    def _extract_features(self, g):
        return np.array([
            np.mean([g.degree(n) for n in g.nodes]),
            np.std([g.nodes[n].get("load", 1) for n in g.nodes]),
            np.mean([d["bandwidth"] for _, _, d in g.edges(data=True)]),
            len(g.nodes)
        ])
