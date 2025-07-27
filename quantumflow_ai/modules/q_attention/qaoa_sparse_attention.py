import numpy as np
from quantumflow_ai.core.logger import get_logger

logger = get_logger("QAOA-Sparse")

def sample_sparse_attention(query: np.ndarray, key: np.ndarray, top_k: int = 3) -> list[tuple[int, int]]:
    relevance = np.dot(query, key.T)
    top_pairs = np.unravel_index(np.argsort(relevance.ravel())[-top_k:], relevance.shape)
    return list(zip(top_pairs[0], top_pairs[1]))
