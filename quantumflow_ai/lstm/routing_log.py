import json
from collections import deque
from datetime import datetime
import os

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

def prepare_lstm_input(logs):
    vecs = []
    for log in logs:
        method_code = 1 if log["method"] == "qaoa" else 0
        input_vec = [method_code] + list(map(float, log["input_matrix"].get("experts", [])))
        vecs.append(input_vec)
    return np.expand_dims(np.array(vecs), axis=0)  # shape: (1, 10, input_dim)
