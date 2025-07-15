import os

def simulate_token_stream(model_graph, num_tokens=8):
    return [{"token_id": i, "route": i % len(model_graph["experts"])} for i in range(num_tokens)]

def load_routing_data(file_path="data/routing_data.csv"):
    """Load routing data from CSV using pandas if available."""
    if not os.path.exists(file_path):
        raise FileNotFoundError("Synthetic data not found. Please run routing_synthetic.py first.")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to load routing data; please install pandas"
        ) from exc

    return pd.read_csv(file_path)
