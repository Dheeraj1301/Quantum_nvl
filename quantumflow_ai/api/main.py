# Auto-generated stub
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from quantumflow_ai.modules.q_routing import optimize_routing, classical_route, simulate_token_stream

app = FastAPI()

class RoutingInput(BaseModel):
    model_graph: dict
    token_stream: list
    use_quantum: bool = True

@app.post("/q-routing")
def route_tokens(data: RoutingInput):
    try:
        if data.use_quantum:
            result = optimize_routing(data.model_graph, data.token_stream)
        else:
            result = classical_route(data.model_graph, data.token_stream)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))