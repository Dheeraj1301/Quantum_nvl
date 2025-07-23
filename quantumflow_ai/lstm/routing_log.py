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

# Persist logs next to this file so the path is stable regardless of cwd
LOG_PATH = os.path.join(os.path.dirname(__file__), "routing_log.json")

# Keep at most the last 10 entries in memory
routing_log = deque(maxlen=10)

def save_log_entry(entry: dict):
    entry["timestamp"] = datetime.utcnow().isoformat()

    # If this is the first entry after a restart, hydrate the deque
    if not routing_log and os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r") as f:
                for item in json.load(f)[-routing_log.maxlen:]:
                    routing_log.append(item)
        except Exception:
            # Ignore malformed log files
            pass

    routing_log.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(list(routing_log), f, indent=2)

def get_last_logs():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []

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
