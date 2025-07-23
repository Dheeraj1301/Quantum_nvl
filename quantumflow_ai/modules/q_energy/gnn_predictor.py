import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, Sequential

class GNNPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 8)
        self.fc = Sequential(Linear(8, 4), ReLU(), Linear(4, 1))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.fc(x).mean()

def prepare_graph_data(job_graph: dict, energy_profile: dict) -> Data:
    job_ids = list(job_graph["jobs"].keys())
    id_to_idx = {jid: idx for idx, jid in enumerate(job_ids)}
    edge_index = []

    for src, targets in job_graph["jobs"].items():
        for tgt in targets:
            edge_index.append([id_to_idx[src], id_to_idx[tgt]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor([[energy_profile[j]] for j in job_ids], dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def predict_energy_with_gnn(job_graph, energy_profile, model_path="modules/q_energy/model/gnn.pt"):
    model = GNNPredictor(in_channels=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    data = prepare_graph_data(job_graph, energy_profile)
    with torch.no_grad():
        return model(data.x, data.edge_index).item()
