# Auto-generated stub
# core/config.py
from pydantic_settings import BaseSettings  # ✅ NEW
from typing import Optional
import random

try:  # Optional heavy deps
    import numpy as _np
except Exception:  # pragma: no cover - optional dep missing
    _np = None

try:
    import torch as _torch
except Exception:  # pragma: no cover - optional dep missing
    _torch = None

try:
    import pennylane as _qml
except Exception:  # pragma: no cover - optional dep missing
    _qml = None

class Settings(BaseSettings):
    quantum_backend: str = "default.qubit"
    use_gpu: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()


def set_global_seed(seed: int | None) -> None:
    """Set random seeds across supported libraries for reproducibility."""
    if seed is None:
        return

    random.seed(seed)

    if _np is not None:
        _np.random.seed(seed)

    if _torch is not None:
        try:
            _torch.manual_seed(seed)
            if _torch.cuda.is_available():  # pragma: no cover - hardware specific
                _torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    if _qml is not None:
        try:
            _qml.numpy.random.seed(seed)
        except Exception:
            pass
