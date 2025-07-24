# tests/test_compression.py
import numpy as np
import pytest

from quantumflow_ai.api.compressor import run_compression
from quantumflow_ai.modules.q_compression.q_autoencoder import (
    QuantumAutoencoder,
    PENNYLANE_AVAILABLE,
)


def test_run_compression_with_dropout_classical():
    data = np.random.rand(5, 4)
    result = run_compression(
        data,
        use_quantum=False,
        use_dropout=True,
        dropout_prob=0.5,
    )
    assert result["mode"] == "classical"


@pytest.mark.skipif(not PENNYLANE_AVAILABLE, reason="PennyLane not installed")
def test_quantum_autoencoder_dropout():
    qae = QuantumAutoencoder(
        n_qubits=2,
        latent_qubits=1,
        use_dropout=True,
        dropout_prob=1.0,
    )
    weights = np.random.randn(3, 2, 3)
    vec = np.random.rand(2)
    result = qae.encode([vec], weights)
    assert len(result[0]) == 1
