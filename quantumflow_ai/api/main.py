# Auto-generated stub
from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
from typing import List
from pydantic import BaseModel
from quantumflow_ai.modules.q_routing import optimize_routing, classical_route

app = FastAPI()

class RoutingInput(BaseModel):
    model_graph: dict
    token_stream: List[dict]
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

@app.post("/q-routing/file-upload")
def route_from_file(file: UploadFile = File(...), use_quantum: bool = True):
    try:
        df = pd.read_csv(file.file)
        token_stream = [{"token_id": int(row["token_id"])} for _, row in df.iterrows()]
        model_graph = {"experts": list(df.columns[1:])}
        if use_quantum:
            result = optimize_routing(model_graph, token_stream)
        else:
            result = classical_route(model_graph, token_stream)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing failed: {e}")
