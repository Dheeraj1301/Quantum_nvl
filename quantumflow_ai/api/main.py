from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
from pydantic import BaseModel
import pandas as pd

from quantumflow_ai.modules.q_routing import optimize_routing, classical_route
from quantumflow_ai.core.logger import get_logger

app = FastAPI()

# 🔓 Allow all cross-origin requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger("QRoutingAPI")

# 🧠 Request schema for JSON-based routing
class RoutingInput(BaseModel):
    model_graph: dict
    token_stream: List[dict]
    use_quantum: bool = True

# 🔍 Simple health/info check
@app.get("/info")
def root():
    return {
        "message": "Quantum Routing API is running.",
        "routes": {
            "POST /q-routing": "Run routing on JSON input",
            "POST /q-routing/file-upload": "Upload .csv to trigger routing"
        }
    }

# 🔄 Route tokens via JSON
@app.post("/q-routing")
def route_tokens(data: RoutingInput):
    try:
        if data.use_quantum:
            logger.info("Routing with QAOA backend")
            result = optimize_routing(data.model_graph, data.token_stream)
        else:
            logger.info("Routing with classical fallback")
            result = classical_route(data.model_graph, data.token_stream)

        return {
            "status": "success",
            "mode": result.get("mode", "classical"),
            "routing_score": result.get("routing_score"),
            "optimized_params": result.get("optimized_params"),
            "assignments": result.get("assignments", []),
            "results": result,
        }

    except Exception as e:
        logger.exception("Routing failed")
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")

# 📁 Route tokens via file upload
@app.post("/q-routing/file-upload")
def route_from_file(file: UploadFile = File(...), use_quantum: bool = True):
    try:
        df = pd.read_csv(file.file)
        if "token_id" not in df.columns:
            raise ValueError("File must include a 'token_id' column.")

        token_stream = [{"token_id": int(row["token_id"])} for _, row in df.iterrows()]
        expert_columns = [col for col in df.columns if col != "token_id"]
        model_graph = {"experts": list(range(len(expert_columns)))}

        logger.info(f"Parsed {len(token_stream)} tokens and {len(expert_columns)} experts from uploaded file.")

        if use_quantum:
            result = optimize_routing(model_graph, token_stream)
        else:
            result = classical_route(model_graph, token_stream)

        return {
            "status": "success",
            "mode": result.get("mode", "classical"),
            "routing_score": result.get("routing_score"),
            "optimized_params": result.get("optimized_params"),
            "assignments": result.get("assignments", []),
            "results": result,
        }

    except Exception as e:
        logger.exception("Failed to process uploaded file")
        raise HTTPException(status_code=400, detail=f"File processing failed: {e}")

# 🧾 Static mount — your frontend HTML, JS, CSS
app.mount("/", StaticFiles(directory="quantumflow_ai/frontend/public", html=True), name="static")
