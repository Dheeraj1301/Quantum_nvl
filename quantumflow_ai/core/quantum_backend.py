# Auto-generated stub
from pennylane import device
from quantumflow_ai.core.config import settings

def get_quantum_device():
    if settings.use_gpu:
        return device("lightning.gpu", wires=8)
    return device(settings.quantum_backend, wires=8)