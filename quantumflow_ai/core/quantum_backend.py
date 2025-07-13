# Auto-generated stub
from pennylane import device
from quantumflow_ai.core.config import settings


def get_quantum_device(wires: int = 8):
    """Return a PennyLane device with the requested number of wires."""
    if settings.use_gpu:
        return device("lightning.gpu", wires=wires)
    return device(settings.quantum_backend, wires=wires)