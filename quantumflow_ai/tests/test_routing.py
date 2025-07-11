# Auto-generated stub
from quantumflow_ai.modules.q_routing import (
    optimize_routing,
    classical_route,
    simulate_token_stream,
)

def test_quantum_routing():
    model_graph = {"experts": [0, 1, 2]}
    token_stream = simulate_token_stream(model_graph, num_tokens=6)
    result = optimize_routing(model_graph, token_stream)
    assert "routing_score" in result
    assert isinstance(result["optimized_params"], list)

def test_classical_routing():
    model_graph = {"experts": [0, 1, 2]}
    token_stream = simulate_token_stream(model_graph, num_tokens=6)
    result = classical_route(model_graph, token_stream)
    assert isinstance(result, list)
    assert "expert" in result[0]
