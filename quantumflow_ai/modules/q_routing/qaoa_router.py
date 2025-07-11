"""Simplified QAOA-based router used for testing.

This module originally relied on the ``pennylane`` quantum computing
library. To keep the test environment lightweight and self contained we
provide a very small stand-in implementation that mimics the expected
behaviour without any external dependencies.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from quantumflow_ai.core.logger import get_logger

logger = get_logger("QAOARouter")


def optimize_routing(model_graph: Dict[str, Any], token_stream: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a dummy routing optimisation result.

    The function generates a pseudo "routing_score" and a list of
    ``optimized_params`` whose length depends on the number of experts in
    the ``model_graph``.  This keeps the interface compatible with a
    potential real implementation while avoiding heavy quantum
    dependencies.
    """
    logger.info("Starting simplified QAOA optimisation")

    n_experts = len(model_graph.get("experts", []))
    optimized_params = [random.random() for _ in range(max(1, n_experts * 2))]

    return {
        "routing_score": random.random(),
        "optimized_params": optimized_params,
    }
