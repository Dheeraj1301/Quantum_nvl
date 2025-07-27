import numpy as np

def quantum_positional_encoding(pos: int, d_model: int) -> np.ndarray:
    return np.array([np.sin(pos / (10000 ** (2 * i / d_model))) for i in range(d_model)])
