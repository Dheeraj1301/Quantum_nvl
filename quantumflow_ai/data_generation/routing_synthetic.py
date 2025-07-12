# quantumflow_ai/data_generation/routing_synthetic.py

import os
import numpy as np
import pandas as pd

def generate_synthetic_routing_data(num_tokens=1024, num_experts=32, sparsity=0.1, noise=0.0):
    """
    Generate a sparse token-to-expert assignment matrix with optional noise.
    Each token can be routed to a few experts based on a binomial pattern.

    Args:
        num_tokens (int): Number of input tokens.
        num_experts (int): Number of experts in the MoE layer.
        sparsity (float): Probability of a connection between token and expert.
        noise (float): Noise to flip 1s randomly to 0s and vice versa.

    Returns:
        pd.DataFrame: Token-to-expert routing matrix.
    """
    matrix = np.random.binomial(1, sparsity, size=(num_tokens, num_experts))

    # Apply binary noise
    if noise > 0:
        noise_mask = np.random.binomial(1, noise, size=matrix.shape)
        matrix = np.logical_xor(matrix, noise_mask).astype(int)

    token_ids = np.arange(num_tokens)
    experts = [f"expert_{i}" for i in range(num_experts)]

    df = pd.DataFrame(matrix, columns=experts)
    df.insert(0, "token_id", token_ids)
    return df

def save_routing_data(df: pd.DataFrame, out_dir="data", format="csv"):
    os.makedirs(out_dir, exist_ok=True)
    if format == "csv":
        df.to_csv(os.path.join(out_dir, "routing_data.csv"), index=False)
    elif format == "json":
        df.to_json(os.path.join(out_dir, "routing_data.json"), orient="records")
    else:
        raise ValueError("Unsupported format")

if __name__ == "__main__":
    df = generate_synthetic_routing_data(num_tokens=2048, num_experts=64, sparsity=0.05, noise=0.02)
    save_routing_data(df)
    print(f"[✔] Generated routing dataset with shape {df.shape} and saved to /data/")
