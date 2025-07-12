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

# Setup wires
if PENNYLANE_AVAILABLE:
    dev = get_quantum_device()
    n_wires = len(dev.wires)
else:
    dev = None
    n_wires = 4  # fallback size

# Hamiltonian builder for MoE routing
def build_moe_hamiltonian(token_stream, num_experts):
    """
    Construct a QAOA Hamiltonian for sparse expert routing.

    Args:
        token_stream (list): List of token dicts, e.g., [{"token_id": 0}, ...]
        num_experts (int): Number of expert slots

    Returns:
        qml.Hamiltonian: Cost Hamiltonian
    """
    if not PENNYLANE_AVAILABLE:
        return None

    coeffs = []
    observables = []

    # Penalize token collisions to the same expert (soft max load balancing)
    for i in range(num_experts):
        coeffs.append(1.0)
        observables.append(qml.PauliZ(i) @ qml.PauliZ(i))

    # Reward sparsity by minimizing active experts (regularization)
    for i in range(num_experts):
        coeffs.append(-0.5)
        observables.append(qml.PauliZ(i))

    return qml.Hamiltonian(coeffs, observables)

# Quantum circuit definition (only if Pennylane is available)
if PENNYLANE_AVAILABLE:
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

    # cost_h will be set inside optimize_routing
    def make_cost_fn(token_stream):
        cost_h = build_moe_hamiltonian(token_stream, n_wires)
        @qml.qnode(dev)
        def cost_fn(params):
            qaoa_ansatz(params)
            return qml.expval(cost_h)
        return cost_fn
else:
    def make_cost_fn(token_stream):
        def cost_fn(params):
            return random.random()
        return cost_fn

def optimize_routing(model_graph: Dict[str, Any], token_stream: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Optimize token-to-expert routing using QAOA or fallback.
    Args:
        model_graph (dict): Graph structure containing expert nodes.
        token_stream (list): List of token identifiers to be routed.

    Returns:
        dict: Routing score and optimized parameters.
    """
    logger.info("Starting QAOA routing optimization")
    n_experts = len(model_graph.get("experts", [])) or n_wires

    if np:
        init_params = np.random.uniform(0, math.pi, n_wires * 2)
    else:
        init_params = [random.uniform(0, math.pi) for _ in range(n_wires * 2)]

    cost_fn = make_cost_fn(token_stream)

    if PENNYLANE_AVAILABLE:
        opt = qml.GradientDescentOptimizer(stepsize=0.4)
        for i in range(20):
            init_params = opt.step(cost_fn, init_params)
            logger.debug(f"Step {i} | Cost: {cost_fn(init_params)}")
        score = float(cost_fn(init_params))
    else:
        score = random.random()

    # --- Data loading and routing example ---
    import pandas as pd
    try:
        df = pd.read_csv("data/routing_data.csv")
        token_stream_csv = [{"token_id": int(row["token_id"])} for _, row in df.iterrows()]
        model_graph_csv = {"experts": list(df.columns[1:])}  # skip token_id
        result_csv = optimize_routing(model_graph_csv, token_stream_csv)
        logger.info(f"CSV Routing Result: {result_csv}")
    except Exception as e:
        logger.warning(f"Could not load routing_data.csv: {e}")

    return {
        "routing_score": score,
        "optimized_params": list(init_params)
    }