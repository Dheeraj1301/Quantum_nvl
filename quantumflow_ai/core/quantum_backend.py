"""Quantum backend helper functions."""
from pennylane import device
from quantumflow_ai.core.config import settings

try:
    import torch
except Exception:  # pragma: no cover - optional dependency missing
    torch = None


def get_quantum_device(wires: int = 8):
    """Return a PennyLane device with an optional GPU backend."""
    backend = settings.quantum_backend

    if settings.use_gpu and torch is not None and getattr(torch, "cuda", None):
        try:
            if torch.cuda.is_available():  # pragma: no cover - hardware specific
                backend = "lightning.gpu"
        except Exception:
            pass

    return device(backend, wires=wires)
