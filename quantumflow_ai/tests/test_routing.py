# quantumflow_ai/tests/test_routing.py
"""Unit tests for QAOA and classical routing logic."""

import os
import sys

# Ensure root project path is discoverable when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from quantumflow_ai.modules.q_routing import (
    optimize_routing,
    classical_route,
    simulate_token_stream,
)

def test_quantum_routing():
    model_graph = {"experts": [0, 1, 2]}
    token_stream = simulate_token_stream(model_graph, num_tokens=6)
    result = optimize_routing(model_graph, token_stream)
    
    assert isinstance(result, dict)
    assert "routing_score" in result
    assert isinstance(result["optimized_params"], list)
    assert isinstance(result["routing_score"], float)

def test_classical_routing():
    model_graph = {"experts": [0, 1, 2]}
    token_stream = simulate_token_stream(model_graph, num_tokens=6)
    result = classical_route(model_graph, token_stream)

    assert isinstance(result, list)
    assert len(result) == 6
    assert "expert" in result[0]
    assert "token_id" in result[0]
