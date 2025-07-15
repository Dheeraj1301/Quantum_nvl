"""QAOA-based routing module with fallback for non-quantum environments.

Uses Pennylane for quantum optimization when available.
Falls back to dummy random optimizer if quantum backend not installed.
"""

from __future__ import annotations
import math
import random
from typing import Any, Dict, List

try:
    import pennylane as qml
    from pennylane import numpy as np
    from quantumflow_ai.core.quantum_backend import get_quantum_device
    PENNYLANE_AVAILABLE = True
except ImportError:
    qml = None
    np = None
    get_quantum_device = None
    PENNYLANE_AVAILABLE = False


from quantumflow_ai.core.logger import get_logger


def _fallback_score(token_stream: List[dict], num_experts: int) -> float:
    """Return a deterministic fallback score based on the token stream."""
    if num_experts <= 0 or not token_stream:
        return 0.0
    total = sum(t.get("token_id", 0) % num_experts for t in token_stream)
    max_total = (num_experts - 1) * len(token_stream)
    if max_total == 0:
        return 0.0
    return total / max_total

logger = get_logger("QAOARouter")

if PENNYLANE_AVAILABLE:
    dev = get_quantum_device()
    n_wires = len(dev.wires)
else:
    dev = None
    n_wires = 0

if PENNYLANE_AVAILABLE:
    def qaoa_ansatz(params):
        for i in range(n_wires):
            qml.Hadamard(wires=i)
        for i in range(len(params)//2):
            for j in range(n_wires - 1):
                qml.CNOT(wires=[j, j+1])
                qml.RZ(params[i], wires=j+1)
                qml.CNOT(wires=[j, j+1])
            for j in range(n_wires):
                qml.RX(params[i + n_wires], wires=j)

    @qml.qnode(dev)
    def cost_fn(params):
        qaoa_ansatz(params)
        return qml.expval(qml.PauliZ(0))

def optimize_routing(model_graph: Dict[str, Any], token_stream: List[Dict[str, Any]], use_deterministic=False) -> Dict[str, Any]:
    """Optimize routing using QAOA if PennyLane is available.

    Falls back to deterministic offline scoring when quantum backend is unavailable.
    """

    # ✅ Always define num_experts first
    num_experts = len(model_graph.get("experts", []))

    if not PENNYLANE_AVAILABLE or num_experts == 0:
        logger.warning("Quantum backend not available or invalid expert count. Using deterministic fallback score.")

        # ✅ Simple load balancing estimate
        expert_loads = [0] * num_experts
        for t in token_stream:
            expert_index = t["token_id"] % num_experts
            expert_loads[expert_index] += 1

        mean_load = sum(expert_loads) / num_experts
        variance = sum((x - mean_load) ** 2 for x in expert_loads) / num_experts
        std_dev = variance ** 0.5

        max_std = mean_load if mean_load > 0 else 1.0  # avoid divide-by-zero
        normalized_score = max(0.0, 1 - std_dev / max_std)

        return {
            "routing_score": round(normalized_score, 4),
            "optimized_params": [],
            "mode": "fallback"
        }

    # ✅ rest of your quantum QAOA logic continues here...


    logger.info("Starting QAOA optimization")
    p_layers = 2
    param_len = 2 * p_layers * n_wires
    init_params = np.random.uniform(0, np.pi, param_len)
    opt = qml.GradientDescentOptimizer(stepsize=0.3)

    prev_cost = float("inf")
    for step in range(25):
        init_params = opt.step(cost_fn, init_params)
        current_cost = cost_fn(init_params)
        logger.debug(f"Step {step} | Cost: {current_cost:.4f}")
        if abs(prev_cost - current_cost) < 1e-4:
            logger.info(f"Converged at step {step}")
            break
        prev_cost = current_cost

    raw_score = float(cost_fn(init_params))
    score = (raw_score + n_wires) / (2 * n_wires)  # Normalize from [-n, n] to [0, 1]

    # ✅ Assign tokens to experts (simple mapping for now)
    assignments = [
        {"token_id": t["token_id"], "expert": i % n_wires}
        for i, t in enumerate(token_stream)
    ]

    return {
        "routing_score": round(score, 4),
        "optimized_params": init_params.tolist(),
        "assignments": assignments,
        "mode": "quantum"
    }
