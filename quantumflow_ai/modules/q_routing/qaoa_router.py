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

def optimize_routing(model_graph, token_stream):
    """Optimize routing using QAOA if PennyLane is available.

    Falls back to a random score when no quantum backend is installed or
    when the provided ``model_graph`` does not contain a valid expert list.
    """

    experts = model_graph.get("experts", [])
    if not isinstance(experts, list) or len(experts) == 0:
        logger.error("Invalid expert count; using fallback router")
        return {"routing_score": random.random(), "optimized_params": []}

    if not PENNYLANE_AVAILABLE:
        logger.warning("Pennylane not installed; using random routing score")
        return {"routing_score": random.random(), "optimized_params": []}

    n_wires = len(experts)
    try:
        dev = get_quantum_device(wires=n_wires)
    except Exception as exc:
        logger.error(f"Quantum backend not available: {exc}; using fallback")
        return {"routing_score": random.random(), "optimized_params": []}

    def qaoa_ansatz(params):
        for i in range(n_wires):
            qml.Hadamard(wires=i)
        for i in range(len(params) // 2):
            for j in range(n_wires - 1):
                qml.CNOT(wires=[j, j + 1])
                qml.RZ(params[i], wires=j + 1)
                qml.CNOT(wires=[j, j + 1])
            for j in range(n_wires):
                qml.RX(params[i + n_wires], wires=j)

    @qml.qnode(dev)
    def cost_fn(params):
        qaoa_ansatz(params)
        return qml.expval(qml.PauliZ(0))

    logger.info("Starting QAOA optimization")
    init_params = np.random.uniform(0, np.pi, n_wires * 2)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    for i in range(20):
        init_params = opt.step(cost_fn, init_params)
        logger.debug(f"Step {i} | Cost: {cost_fn(init_params)}")

    score = float(cost_fn(init_params))
    return {
        "routing_score": score,
        "optimized_params": list(init_params)
    }