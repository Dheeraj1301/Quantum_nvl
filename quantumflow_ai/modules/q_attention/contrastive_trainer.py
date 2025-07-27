import numpy as np
from quantumflow_ai.core.logger import get_logger

logger = get_logger("ContrastiveQML")

def contrastive_loss(real: np.ndarray, corrupted: np.ndarray) -> float:
    pos_score = np.dot(real, real)
    neg_score = np.dot(real, corrupted)
    return max(0, 1 - pos_score + neg_score)
