import json
from collections import deque
from datetime import datetime
import os
from typing import List

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    np = None
    NUMPY_AVAILABLE = False

LOG_PATH = "routing_log.json"
routing_log = deque(maxlen=10)

def save_log_entry(entry: dict):
    entry["timestamp"] = datetime.utcnow().isoformat()
    routing_log.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(list(routing_log), f, indent=2)

def get_last_logs():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        return json.load(f)

def prepare_lstm_input(logs: List[dict]):
    """Return 3D numpy array suitable for LSTM training or inference."""

    feature_vecs: List[List[float]] = []
    for log in logs[-10:]:
        method_code = 1.0 if log.get("method") == "qaoa" else 0.0
        num_experts = float(len(log.get("input_matrix", {}).get("experts", [])))
        token_count = float(log.get("token_count", len(log.get("output_matrix", []))))
        energy = float(log.get("energy", 0.0))

        feature_vecs.append([method_code, num_experts, token_count, energy])

    if not NUMPY_AVAILABLE:
        return []

    if not feature_vecs:
        return np.empty((1, 0, 4))

    arr = np.array(feature_vecs, dtype=float)
    return np.expand_dims(arr, axis=0)
