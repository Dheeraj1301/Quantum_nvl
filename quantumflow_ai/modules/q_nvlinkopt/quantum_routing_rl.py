import torch
import torch.nn as nn
import random

class QuantumRoutingAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def select_action(self, state):
        logits = self.model(state)
        return torch.argmax(logits).item()

    def update(self, states, actions, rewards):
        loss_fn = nn.CrossEntropyLoss()
        logits = self.model(states)
        loss = loss_fn(logits, actions)
        loss.backward()
