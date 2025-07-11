# Auto-generated stub
def simulate_token_stream(model_graph, num_tokens=8):
    return [{"token_id": i, "route": i % len(model_graph["experts"])} for i in range(num_tokens)]
