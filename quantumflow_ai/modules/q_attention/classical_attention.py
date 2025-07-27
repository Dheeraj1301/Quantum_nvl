# Auto-generated stub
import numpy as np

def classical_linear_attention(query, key, value):
    return np.matmul(np.matmul(query, key.T), value) / np.sqrt(query.shape[-1])
