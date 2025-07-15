# Auto-generated stub
from quantumflow_ai.core.logger import get_logger
from quantumflow_ai.modules.routing_advisor import record_routing_run
logger = get_logger("ClassicalRouter")

def classical_route(model_graph, token_stream):
    logger.info("Running classical fallback router")
    expert_count = len(model_graph["experts"])
    assignments = [
        {"token_id": t["token_id"], "expert": t["token_id"] % expert_count}
        for t in token_stream
    ]

    record_routing_run(
        model_graph,
        token_stream,
        {"routing_score": 0.0, "assignments": assignments},
        use_quantum=False,
    )

    return assignments
