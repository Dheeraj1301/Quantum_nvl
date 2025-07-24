import numpy as np
import pytest
from quantumflow_ai.api.compressor import run_compression, validate_compression_config
from quantumflow_ai.modules.q_compression.q_autoencoder import (
    QuantumAutoencoder,
    PENNYLANE_AVAILABLE,
)


def test_prediction_length_and_values():
    data = np.random.rand(5, 4)
    result = run_compression(data, use_quantum=False, predict_first=True)
    preds = result.get("predictions")
    assert isinstance(preds, list)
    assert len(preds) == len(data)
    assert all(p in (0, 1) for p in preds)


def test_validate_config_conflicts():
    cfg = {
        "use_denoiser": True,
        "use_dropout": True,
        "compression_mode": "hybrid",
        "enable_pruning": True,
        "predict_compressibility": False,
    }
    clean = validate_compression_config(cfg)
    assert clean["use_denoiser"] is False
    assert clean["use_dropout"] is False
    assert clean["compression_mode"] == "qml"
    assert clean.get("predict_first", False) is False


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
