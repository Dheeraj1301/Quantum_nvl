import numpy as np
from quantumflow_ai.core.logger import get_logger

logger = get_logger("Q-SVD-Pruner")

def prune_heads_with_qsvd(head_matrices: list[np.ndarray], threshold=0.1):
    pruned_heads = []
    for i, head in enumerate(head_matrices):
        u, s, vh = np.linalg.svd(head)
        if s[0] >= threshold:
            pruned_heads.append(head)
        else:
            logger.info(f"Pruned head {i} with singular value {s[0]}")
    return pruned_heads
