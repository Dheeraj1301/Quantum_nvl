from __future__ import annotations

"""Utility for analysing routing logs and providing tuning advice."""

from typing import Any, Dict

from quantumflow_ai.lstm.routing_log import (
    save_log_entry,
    get_last_logs,
    prepare_lstm_input,
    NUMPY_AVAILABLE,
    np,
)
from quantumflow_ai.lstm.lstm_model import load_or_train_model


def record_routing_run(
    model_graph: Dict[str, Any],
    token_stream: Any,
    result: Dict[str, Any],
    use_quantum: bool,
) -> None:
    """Persist the outcome of a routing run for later analysis."""
    entry = {
        "input_matrix": model_graph,
        "output_matrix": result.get("assignments", []),
        "energy": result.get("routing_score", 0.0),
        "method": "qaoa" if use_quantum else "classical",
        "token_count": len(token_stream),
    }
    save_log_entry(entry)


def advise_routing_strategy() -> Dict[str, Any]:
    """Return heuristic suggestions based on the last logged runs."""
    logs = get_last_logs()
    if len(logs) < 2:
        return {"note": "Not enough data for suggestion"}

    X = prepare_lstm_input(logs)
    if not X or (hasattr(X, "size") and X.size == 0):
        return {"note": "LSTM features unavailable"}

    y = None
    if NUMPY_AVAILABLE:
        y = np.array([log.get("energy", 0.0) for log in logs[-X.shape[1]:]], dtype=float)
        y = y.reshape(1, -1, 1)

    model = load_or_train_model(X, y)
    prediction = model.predict(X)[0][0]
    suggest_quantum = bool(prediction >= 0.5)

    last_assignments = logs[-1].get("output_matrix", [])
    expert_loads = {}
    for item in last_assignments:
        idx = item.get("expert")
        expert_loads[idx] = expert_loads.get(idx, 0) + 1

    if expert_loads:
        load_values = list(expert_loads.values())
        imbalance = max(load_values) - min(load_values)
    else:
        imbalance = 0

    return {
        "suggested_use_quantum": suggest_quantum,
        "imbalance": imbalance,
        "raw_prediction": float(prediction),
    }
