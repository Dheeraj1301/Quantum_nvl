# Auto-generated stub
# modules/q_decompression/qft_decoder.py

import logging
import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as qnp
    QML_AVAILABLE = True
except ImportError:
    QML_AVAILABLE = False

logger = logging.getLogger("QFTDecoder")

class QFTDecoder:
    def __init__(self, num_qubits: int = 4, use_quantum: bool = True, learnable: bool = False, amplitude_estimate: bool = False):
        self.num_qubits = num_qubits
        self.use_quantum = use_quantum and QML_AVAILABLE
        self.learnable = learnable
        self.amplitude_estimate = amplitude_estimate
        self.dev = qml.device("default.qubit", wires=num_qubits) if self.use_quantum else None

        if self.learnable:
            self.thetas = qnp.array([0.1] * self.num_qubits, requires_grad=True)

    def parameterized_qft(self):
        for i in range(self.num_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(self.thetas[i], wires=i)
            for j in range(i + 1, self.num_qubits):
                qml.ctrl(qml.RZ, control=j)(np.pi / 2 ** (j - i), wires=i)

    def decode(self, compressed_input: np.ndarray) -> np.ndarray:
        logger.info(f"Running QFT decoding (quantum={self.use_quantum}, learnable={self.learnable}, amp_est={self.amplitude_estimate})")

        if not self.use_quantum:
            logger.warning("Quantum backend unavailable. Using classical IFFT.")
            return np.fft.ifft(compressed_input)

        @qml.qnode(self.dev)
        def circuit():
            qml.AmplitudeEmbedding(compressed_input, wires=range(self.num_qubits), normalize=True)
            if self.learnable:
                self.parameterized_qft()
            else:
                for i in range(self.num_qubits):
                    qml.Hadamard(wires=i)
                    for j in range(i+1, self.num_qubits):
                        qml.ctrl(qml.RZ, control=j)(np.pi / 2 ** (j - i), wires=i)

            return [qml.probs(wires=i) if self.amplitude_estimate else qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return np.array(circuit())
