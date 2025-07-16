# train_gnn_model.py

import torch
import json
import os
import sys
from pathlib import Path

# Allow running this script directly from the notebooks folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantumflow_ai.modules.q_energy.gnn_predictor import GNNPredictor, prepare_graph_data

def train_gnn(
    profiles=["a100", "h100", "gb200"],
    data_dir="quantumflow_ai/notebooks/profiles",
    model_out="modules/q_energy/model/gnn.pt",
):
    model = GNNPredictor(in_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    dataset = []
    for profile in profiles:
        path = os.path.join(data_dir, f"{profile}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            items = json.load(f)
        for d in items:
            g = prepare_graph_data(d["graph"], d["energy_profile"])
            g.y = torch.tensor([d["cost"]], dtype=torch.float)
            dataset.append(g)

    if not dataset:
        raise RuntimeError("No graph data loaded for training")

    for epoch in range(50):
        total_loss = 0
        for sample in dataset:
            optimizer.zero_grad()
            out = model(sample.x, sample.edge_index)
            loss = loss_fn(out, sample.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    # Ensure the directory for the model output exists. ``torch.save`` will
    # raise a ``RuntimeError`` if the parent directory is missing. This can
    # happen when running the training command in a fresh repository checkout.
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"✅ GNN model saved to {model_out}")


if __name__ == "__main__":
    train_gnn()
