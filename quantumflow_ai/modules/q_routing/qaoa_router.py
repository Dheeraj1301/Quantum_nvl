"""QAOA-based routing module with fallback for non-quantum environments.

Uses Pennylane for quantum optimization when available.
Falls back to dummy random optimizer if quantum backend not installed.
"""

from __future__ import annotations
import random
import math
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
logger = get_logger("QAOARouter")

# ================================
# Cost Hamiltonian Construction
# ================================
if PENNYLANE_AVAILABLE:
    def build_cost_hamiltonian(num_wires: int):
        """Simple cost: encourage load balancing across experts."""
        coeffs = [1.0] * num_wires
        observables = [qml.PauliZ(i) for i in range(num_wires)]
        return qml.Hamiltonian(coeffs, observables)

    def qaoa_ansatz(params, wires):
        """Basic layered QAOA ansatz."""
        for i in wires:
            qml.Hadamard(wires=i)

        p = len(params) // (2 * len(wires))
        for layer in range(p):
            gamma_idx = layer * len(wires)
            beta_idx = gamma_idx + len(wires)
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i+1]])
                qml.RZ(params[gamma_idx + i], wires=wires[i+1])
                qml.CNOT(wires=[wires[i], wires[i+1]])
            for i in range(len(wires)):
                qml.RX(params[beta_idx + i], wires=i)

# ================================
# Main Routing Function
# ================================
def optimize_routing(model_graph: Dict[str, Any], token_stream: List[Dict[str, Any]], use_deterministic=False) -> Dict[str, Any]:
    """Optimize routing using QAOA if PennyLane is available.

    Falls back to a random score when no quantum backend is installed.
    """
    num_experts = len(model_graph.get("experts", []))

    if not PENNYLANE_AVAILABLE or num_experts == 0:
        logger.warning("Quantum backend not available or invalid expert count. Fallback mode.")
        return {
            "routing_score": round(random.random(), 4),
            "optimized_params": [],
            "assignments": [
                {"token_id": t["token_id"], "expert": t["token_id"] % max(1, num_experts)}
                for t in token_stream
            ],
            "mode": "fallback"
        }

    n_wires = num_experts
    wires = list(range(n_wires))
    cost_h = build_cost_hamiltonian(n_wires)

    dev = get_quantum_device(wires=n_wires)
    if not dev:
        logger.warning("Failed to acquire quantum device. Falling back.")
        return {
            "routing_score": round(random.random(), 4),
            "optimized_params": [],
            "assignments": [
                {"token_id": t["token_id"], "expert": t["token_id"] % n_wires}
                for t in token_stream
            ],
            "mode": "fallback"
        }

    if use_deterministic and np:
        np.random.seed(1234)

    @qml.qnode(dev)
    def cost_fn(params):
        qaoa_ansatz(params, wires)
        return qml.expval(cost_h)

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
