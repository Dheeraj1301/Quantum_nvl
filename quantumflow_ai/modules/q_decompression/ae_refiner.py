# modules/q_decompression/ae_refiner.py

import numpy as np
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("AERefiner")

class SimpleAutoencoder(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(dim//2, dim), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AERefiner:
    def __init__(self, dim: int = 16):
        self.model = SimpleAutoencoder(dim=dim)
        self.model.eval()

    def refine(self, x: np.ndarray) -> np.ndarray:
        logger.info("Refining QFT output using autoencoder")
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            refined = self.model(x_tensor).numpy()
        return refined
