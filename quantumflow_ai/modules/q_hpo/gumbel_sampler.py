# quantumflow_ai/modules/q_hpo/gumbel_sampler.py
import torch
import torch.nn.functional as F
from quantumflow_ai.core.logger import get_logger

logger = get_logger("GumbelSampler")

class GumbelSoftmaxSampler:
    def __init__(self, categories: dict, temperature: float = 1.0):
        self.categories = categories
        self.temperature = temperature

    def sample(self):
        sampled = {}
        for key, values in self.categories.items():
            logits = torch.randn(len(values))
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            index = torch.argmax(probs).item()
            sampled[key] = values[index]
        return sampled
