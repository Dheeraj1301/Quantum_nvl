# modules/q_decompression/lstm_enhancer.py

import numpy as np
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("LSTMEnhancer")

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)

class LSTMEnhancer:
    def __init__(self):
        self.model = LSTMModel()
        self.model.eval()

    def enhance(self, x: np.ndarray) -> np.ndarray:
        logger.info("Enhancing input using LSTM temporal model")
        x_tensor = torch.tensor(x.reshape(1, -1, 1), dtype=torch.float32)
        with torch.no_grad():
            enhanced = self.model(x_tensor).squeeze(0).squeeze(-1).numpy()
        return enhanced
