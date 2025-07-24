"""Simple Variational Quantum Classifier used to predict compressibility."""

try:
    import pennylane as qml
    from pennylane import numpy as np
    PENNYLANE_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    qml = None  # type: ignore
    np = None
    PENNYLANE_AVAILABLE = False

from quantumflow_ai.core.quantum_backend import get_quantum_device
from quantumflow_ai.core.config import set_global_seed

from sklearn.linear_model import LogisticRegression

class VQCClassifier:
    """Train a basic VQC on binary labels with logistic fallback."""

    def __init__(self, n_qubits: int, seed: int | None = 42):
        self.n_qubits = n_qubits
        self.seed = seed
        self.weights = None
        if PENNYLANE_AVAILABLE:
            set_global_seed(seed)
            self.device = get_quantum_device(wires=n_qubits)
            self.qnode = qml.QNode(self._circuit, self.device, interface="autograd")
        else:  # pragma: no cover - non-quantum fallback
            self.model = LogisticRegression()

    def _circuit(self, inputs, weights):  # pragma: no cover - executed in quantum tests
        qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        return qml.expval(qml.PauliZ(0))

    def _cost(self, weights, xs, ys):  # pragma: no cover - executed in quantum tests
        loss = 0.0
        for x, y in zip(xs, ys):
            out = self.qnode(x, weights)
            prob = (1 - out) / 2
            loss += (prob - y) ** 2
        return loss / len(xs)

    def train(self, xs, ys, steps: int = 30, lr: float = 0.2):
        if not PENNYLANE_AVAILABLE:
            self.model.fit(xs, ys)
            return None

        weights = np.random.randn(3, self.n_qubits, 3, requires_grad=True)
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(steps):
            weights = opt.step(lambda w: self._cost(w, xs, ys), weights)
        self.weights = weights
        return weights

    def predict(self, xs):
        if PENNYLANE_AVAILABLE and self.weights is not None:
            preds = []
            for x in xs:
                out = self.qnode(x, self.weights)
                prob = (1 - out) / 2
                preds.append(1 if prob >= 0.5 else 0)
            return preds

        return self.model.predict(xs).tolist()
