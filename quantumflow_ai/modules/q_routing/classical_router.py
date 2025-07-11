# Auto-generated stub
from core.logger import get_logger
logger = get_logger("ClassicalRouter")

def classical_route(model_graph, token_stream):
    logger.info("Running classical fallback router")
    expert_count = len(model_graph["experts"])
    return [
        {"token_id": t["token_id"], "expert": t["token_id"] % expert_count}
        for t in token_stream
    ]
