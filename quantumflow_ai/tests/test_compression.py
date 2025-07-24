import numpy as np
from quantumflow_ai.api.compressor import run_compression, validate_compression_config


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
