"""LSTM-based failure predictor used by the pipeline manager."""
from __future__ import annotations

from typing import List

try:
    from tensorflow.keras.models import load_model  # pragma: no cover - optional
    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    TENSORFLOW_AVAILABLE = False
    load_model = None

MODEL_PATH = __name__.replace(".", "/") + "_fp.keras"


def predict_retry(features: List[float]) -> float:
    """Return a retry success probability based on feature vector.

    When TensorFlow is unavailable, a light-weight heuristic is used.
    """
    if TENSORFLOW_AVAILABLE:
        try:
            model = load_model(MODEL_PATH)
            import numpy as np

            arr = np.array(features, dtype=float).reshape(1, -1, len(features))
            pred = model.predict(arr, verbose=0)
            return float(pred[0][0])
        except Exception:  # pragma: no cover - any load/predict issue
            pass

    # Simple heuristic fallback
    score = 1.0 - min(1.0, sum(features) / (len(features) * 10.0))
    return score

