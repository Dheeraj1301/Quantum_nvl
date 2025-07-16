# train_meta_model.py

import json
import os
from quantumflow_ai.modules.q_energy.meta_scheduler import MetaScheduler

def train_meta_from_profiles(profiles, data_dir="notebooks/profiles"):
    all_X, all_y = [], []

    for profile in profiles:
        path = os.path.join(data_dir, f"{profile}.json")
        if not os.path.exists(path):
            print(f"⚠️ Skipping {profile}, file not found.")
            continue
        with open(path) as f:
            samples = json.load(f)
        for d in samples:
            all_X.append(d["features"])
            cost = d["cost"]
            # Basic heuristic
            if cost < 25:
                all_y.append("qbm")
            elif cost < 60:
                all_y.append("classical")
            else:
                all_y.append("qaoa")

    if not all_X:
        raise RuntimeError("No training data found.")

    meta = MetaScheduler()
    meta.train(all_X, all_y)
    print("✅ Meta strategy model trained and saved.")


if __name__ == "__main__":
    train_meta_from_profiles(["a100", "h100", "gb200"])
