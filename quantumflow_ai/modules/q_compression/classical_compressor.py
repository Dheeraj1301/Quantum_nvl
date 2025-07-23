# Auto-generated stub
from sklearn.decomposition import PCA
from typing import Any
import numpy as np

class ClassicalCompressor:
    def __init__(self, n_components: int = 4):
        self.model = PCA(n_components=n_components)

    def fit(self, X: np.ndarray):
        self.model.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.inverse_transform(X)
