"""Optional latent space denoiser used after quantum compression."""

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None
    TORCH_AVAILABLE = False

from quantumflow_ai.core.logger import get_logger

logger = get_logger("LatentDenoiser")


from typing import Union


class LatentDenoiser(nn.Module if TORCH_AVAILABLE else object):  # pragma: no cover - optional
    """Simple autoencoder to denoise latent vectors."""

    def __init__(self, latent_dim: int):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for LatentDenoiser")
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, x: "torch.Tensor"):  # type: ignore[name-defined]
        return self.model(x)

    def train_denoiser(
        self, vectors: Union[list, "torch.Tensor"], epochs: int = 50, lr: float = 1e-3
    ) -> float:
        """Train the autoencoder on latent vectors."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for LatentDenoiser")

        if not isinstance(vectors, torch.Tensor):
            data = torch.tensor(vectors, dtype=torch.float32)
        else:
            data = vectors.float()

        # Ensure the tensor has the expected latent dimension
        if data.dim() == 1:
            data = data.unsqueeze(1)
        expected = self.model[0].in_features
        if data.numel() == 0 or data.shape[-1] != expected:
            raise ValueError(
                f"LatentDenoiser expected vectors of size {expected}, got {tuple(data.shape)}"
            )

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for _ in range(epochs):
            opt.zero_grad()
            out = self.forward(data)
            loss = criterion(out, data)
            loss.backward()
            opt.step()
        logger.info(f"[Denoiser] Final training loss: {loss.item():.4f}")
        return float(loss.item())
