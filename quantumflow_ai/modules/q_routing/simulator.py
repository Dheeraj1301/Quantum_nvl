import os
import pandas as pd

def simulate_token_stream(model_graph, num_tokens=8):
    return [{"token_id": i, "route": i % len(model_graph["experts"])} for i in range(num_tokens)]

def load_routing_data(file_path="data/routing_data.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError("Synthetic data not found. Please run routing_synthetic.py first.")
    return pd.read_csv(file_path)
