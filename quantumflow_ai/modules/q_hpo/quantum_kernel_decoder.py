# quantumflow_ai/modules/q_hpo/quantum_kernel_decoder.py
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from quantumflow_ai.core.logger import get_logger

logger = get_logger("QuantumKernelDecoder")

class KernelMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

class QuantumKernelDecoder:
    def __init__(self, config_vectors):
        self.config_vectors = config_vectors
        self.kernel_matrix = self.compute_kernel_matrix()
        self.model = KernelMLP(input_dim=len(config_vectors))

    def compute_kernel_matrix(self):
        return qml.kernels.projected_distance(self.config_vectors, scale=0.5)

    def train(self, loss_values, epochs=100):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        x = torch.tensor(self.kernel_matrix, dtype=torch.float32)
        y = torch.tensor(loss_values, dtype=torch.float32).unsqueeze(1)

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        return self.model(x).detach().numpy()
