# quantumflow_ai/modules/q_hpo/meta_lstm_predictor.py
import torch
import torch.nn as nn
from quantumflow_ai.core.logger import get_logger

logger = get_logger("MetaLSTM")

class MetaLSTMPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

    def predict_next(self, sequence):
        self.eval()
        with torch.no_grad():
            return self.forward(torch.tensor(sequence).float().unsqueeze(0)).item()
