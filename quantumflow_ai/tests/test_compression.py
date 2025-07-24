import numpy as np
from quantumflow_ai.api.compressor import run_compression


def test_prediction_length_and_values():
    data = np.random.rand(5, 4)
    result = run_compression(data, use_quantum=False, predict_first=True)
    preds = result.get("predictions")
    assert isinstance(preds, list)
    assert len(preds) == len(data)
    assert all(p in (0, 1) for p in preds)
