import os
import sys
import pandas as pd

# ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from quantumflow_ai.modules.q_routing import optimize_routing, classical_route
from quantumflow_ai.modules.q_routing.simulator import load_routing_data


def evaluate_classical_router(df):
    num_experts = len([c for c in df.columns if c.startswith('expert_')])
    model_graph = {"experts": list(range(num_experts))}
    token_stream = [{"token_id": i} for i in df['token_id']]
    predictions = classical_route(model_graph, token_stream)

    correct = 0
    for pred in predictions:
        token_id = pred['token_id']
        expert = pred['expert']
        if df.loc[token_id, f'expert_{expert}'] == 1:
            correct += 1
    accuracy = correct / len(predictions)
    return accuracy


def evaluate_quantum_router(df):
    num_experts = len([c for c in df.columns if c.startswith('expert_')])
    model_graph = {"experts": list(range(num_experts))}
    token_stream = [{"token_id": i} for i in df['token_id']]
    result = optimize_routing(model_graph, token_stream)
    return result['routing_score']


if __name__ == "__main__":
    df = load_routing_data(os.path.join(project_root, 'data', 'routing_data.csv'))

    classical_acc = evaluate_classical_router(df)
    print(f"Classical router accuracy: {classical_acc:.2%}")

    quantum_score = evaluate_quantum_router(df)
    print(f"Quantum router score: {quantum_score:.4f}")
