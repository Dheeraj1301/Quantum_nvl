import numpy as np
from quantumflow_ai.core.logger import get_logger

logger = get_logger("VQC-Reweight")

try:
    import pennylane as qml
    from pennylane import numpy as qnp
except ImportError:
    qml, qnp = None, None

def vqc_reweight(QK_concat: np.ndarray, wires: int = 4):
    if qml is None:
        logger.warning("PennyLane not found. Skipping VQC.")
        return np.ones((QK_concat.shape[0],))

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit(params, x):
        qml.AngleEmbedding(x, wires=range(wires))
        qml.StronglyEntanglingLayers(params, wires=range(wires))
        return qml.expval(qml.PauliZ(0))

    params = qnp.random.randn(1, wires, 3)
    weights = [circuit(params, q[:wires]) for q in QK_concat]
    return np.array(weights)
