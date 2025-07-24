"""Quantum autoencoder implementation used for Q-Compression."""

try:
    import pennylane as qml
    from pennylane import numpy as np
    PENNYLANE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    qml = None  # type: ignore
    np = None
    PENNYLANE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

from quantumflow_ai.core.quantum_backend import get_quantum_device
from quantumflow_ai.core.logger import get_logger
from quantumflow_ai.core.config import set_global_seed

logger = get_logger("QuantumAutoencoder")

class QuantumAutoencoder:
    def __init__(
        self,
        n_qubits: int,
        latent_qubits: int,
        seed: int | None = 42,
        *,
        noise: bool = False,
        noise_level: float = 0.0,
        use_dropout: bool = False,
        dropout_prob: float = 0.0,
    ):
        if not PENNYLANE_AVAILABLE:
            raise RuntimeError("PennyLane is required for QuantumAutoencoder")

        set_global_seed(seed)

        self.n_qubits = n_qubits
        self.latent_qubits = latent_qubits
        self.noise = noise
        # Clamp noise level to a sane range so a noisy frontend cannot break the circuit
        self.noise_level = max(0.0, min(noise_level, 0.3))
        self.use_dropout = use_dropout
        self.dropout_prob = max(0.0, min(dropout_prob, 1.0))
        if noise and self.noise_level > 0:
            # default.qubit does not support noisy operations. Switch to a
            # noise-capable device when noise injection is requested.
            self.device = qml.device("default.mixed", wires=n_qubits)
        else:
            self.device = get_quantum_device(wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.device, interface="autograd")

    def _circuit(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))

        if self.noise and self.noise_level > 0 or self.use_dropout:
            # Apply each entangling layer individually so noise or dropout can be injected
            for idx in range(len(weights)):
                if self.use_dropout and np.random.random() < self.dropout_prob:
                    for i in range(self.n_qubits):
                        qml.Identity(wires=i)
                else:
                    qml.templates.StronglyEntanglingLayers(
                        weights[idx : idx + 1], wires=range(self.n_qubits)
                    )
                if self.noise and self.noise_level > 0:
                    for i in range(self.n_qubits):
                        qml.DepolarizingChannel(self.noise_level, wires=i)
        else:
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

        return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_qubits)]

    def cost_fn(self, weights: np.ndarray, inputs: list[np.ndarray]):
        loss = 0.0
        for x in inputs:
            output = np.asarray(self.qnode(x, weights))
            if output.shape != (self.latent_qubits,):
                raise ValueError(
                    f"Unexpected qnode output shape {output.shape};"
                    f" expected ({self.latent_qubits},)"
                )
            loss += np.sum((x[: self.latent_qubits] - output) ** 2)
        return loss / len(inputs)

    def train(self, inputs: list[np.ndarray], steps: int = 100, lr: float = 0.1):
        weights = np.random.randn(3, self.n_qubits, 3, requires_grad=True)
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        config_hash = hash(f"{self.n_qubits}_{self.latent_qubits}_{steps}")
        for i in range(steps):
            weights = opt.step(lambda w: self.cost_fn(w, inputs), weights)
            if i % 10 == 0:
                loss = self.cost_fn(weights, inputs)
                logger.info(f"[QAE-{config_hash}] Step {i}: Loss = {loss:.4f}")
        return weights

    def encode(self, inputs: list[np.ndarray], weights: np.ndarray):
        """Encode input vectors using the trained quantum circuit."""
        encoded = []
        for x in inputs:
            output = np.asarray(self.qnode(x, weights))
            encoded.append(output.tolist())
        return encoded

    def prune_qubits(
        self,
        encoded: list[list[float]] | list[np.ndarray],
        *,
        threshold: float = 0.01,
    ) -> tuple[list[int], list[list[float]]]:
        """Prune output qubits with variance below ``threshold``.

        Parameters
        ----------
        encoded:
            Encoded latent vectors returned by :func:`encode`.
        threshold:
            Minimum variance required to keep a qubit.

        Returns
        -------
        tuple
            Indices of qubits kept and the pruned encoded vectors.
        """

        arr = np.asarray(encoded)
        variances = np.var(arr, axis=0)
        keep = [i for i, var in enumerate(variances) if var >= threshold]
        pruned = arr[:, keep]
        return keep, pruned.tolist()

    def save_weights(self, weights: np.ndarray, path: str) -> None:
        """Persist trained weights to ``path``."""
        np.save(path, weights)

    def load_weights(self, path: str) -> np.ndarray:
        """Load weights previously stored via :func:`save_weights`."""
        return np.load(path)


class QuantumAutoencoderTorch(
    torch.nn.Module if TORCH_AVAILABLE else object
):  # pragma: no cover - optional
    """PyTorch wrapper around :class:`QuantumAutoencoder`."""

    def __init__(self, qae: 'QuantumAutoencoder', weights: np.ndarray):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for QuantumAutoencoderTorch")
        super().__init__()
        self.qae = qae
        self.weights = weights

    def forward(self, x: 'torch.Tensor'):  # type: ignore[name-defined]
        return torch.tensor(
            self.qae.encode([x.detach().cpu().numpy()], self.weights)[0]
        )

