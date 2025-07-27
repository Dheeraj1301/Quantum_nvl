import numpy as np
from quantumflow_ai.core.logger import get_logger

logger = get_logger("QKA-Kernel")

try:
    import pennylane as qml
    from pennylane.kernels import squared_fidelity
except ImportError:
    qml = None

def quantum_kernel_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    if qml is None:
        logger.warning("PennyLane kernel module not available. Falling back.")
        return None

    logger.info("Computing quantum kernel attention matrix...")
    kernel_matrix = np.array([[squared_fidelity(q, k) for k in key] for q in query])
    kernel_matrix = np.exp(kernel_matrix)  # Optional softmax scaling
    attention_output = np.matmul(kernel_matrix, value)
    return attention_output
