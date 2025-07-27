import numpy as np

def scale_and_pad(tensor: np.ndarray, target_len: int) -> np.ndarray:
    current = tensor.shape[1]
    if current < target_len:
        return np.pad(tensor, ((0, 0), (0, target_len - current)), mode='constant')
    return tensor[:, :target_len]
